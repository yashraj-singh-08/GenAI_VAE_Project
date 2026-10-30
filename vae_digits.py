"""
Handwritten Digit Generation with a Variational Autoencoder (VAE)
Gen AI (CS3ED13) Mini Project - implemented from scratch in NumPy.
Dataset: scikit-learn 'digits' (1797 images, 8x8 grayscale) - no download needed.
Run:  python vae_digits.py
"""
import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
EPS = 1e-7


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class VAE:
    """Encoder 64 -> 48 (tanh) -> (mu, log-var); decoder z -> 48 (tanh) -> 64 (sigmoid)."""

    def __init__(self, in_dim=64, hid=48, z_dim=2, seed=0):
        r = np.random.RandomState(seed)
        s = lambda a, b: r.randn(a, b) * np.sqrt(2.0 / a)
        self.z_dim = z_dim
        self.p = {"W1": s(in_dim, hid), "b1": np.zeros(hid),
                  "Wmu": s(hid, z_dim) * 0.1, "bmu": np.zeros(z_dim),
                  "Wlv": s(hid, z_dim) * 0.1, "blv": np.zeros(z_dim),
                  "W3": s(z_dim, hid), "b3": np.zeros(hid),
                  "W4": s(hid, in_dim), "b4": np.zeros(in_dim)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0
        self.rng = np.random.RandomState(seed + 1)

    def encode(self, x):
        p = self.p
        h = np.tanh(x @ p["W1"] + p["b1"])
        return h, h @ p["Wmu"] + p["bmu"], h @ p["Wlv"] + p["blv"]

    def decode(self, z):
        p = self.p
        h2 = np.tanh(z @ p["W3"] + p["b3"])
        return h2, sigmoid(h2 @ p["W4"] + p["b4"])

    def loss_and_grads(self, x, beta=1.0):
        p, n = self.p, x.shape[0]
        h, mu, lv = self.encode(x)
        eps = self.rng.randn(*mu.shape)
        std = np.exp(0.5 * lv)
        z = mu + std * eps                       # reparameterisation trick
        h2, xr = self.decode(z)
        xr_c = np.clip(xr, EPS, 1 - EPS)
        rec = -np.sum(x * np.log(xr_c) + (1 - x) * np.log(1 - xr_c)) / n
        kl = -0.5 * np.sum(1 + lv - mu ** 2 - np.exp(lv)) / n
        loss = rec + beta * kl
        # ---- manual back-propagation ----
        d_logit = (xr - x) / n
        g = {}
        g["W4"] = h2.T @ d_logit
        g["b4"] = d_logit.sum(0)
        d_h2 = (d_logit @ p["W4"].T) * (1 - h2 ** 2)
        g["W3"] = z.T @ d_h2
        g["b3"] = d_h2.sum(0)
        d_z = d_h2 @ p["W3"].T
        d_mu = d_z + beta * mu / n
        d_lv = d_z * eps * 0.5 * std + beta * 0.5 * (np.exp(lv) - 1) / n
        g["Wmu"] = h.T @ d_mu
        g["bmu"] = d_mu.sum(0)
        g["Wlv"] = h.T @ d_lv
        g["blv"] = d_lv.sum(0)
        d_h = (d_mu @ p["Wmu"].T + d_lv @ p["Wlv"].T) * (1 - h ** 2)
        g["W1"] = x.T @ d_h
        g["b1"] = d_h.sum(0)
        return loss, rec, kl, g

    def adam(self, g, lr=3e-3, b1=0.9, b2=0.999):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def fit(self, Xtr, Xte, epochs=300, bs=64, beta=1.0):
        hist = {"train": [], "test": [], "rec": [], "kl": []}
        for ep in range(epochs):
            idx = self.rng.permutation(len(Xtr))
            for i in range(0, len(Xtr), bs):
                _, _, _, g = self.loss_and_grads(Xtr[idx[i:i + bs]], beta)
                self.adam(g)
            tr, rec, kl, _ = self.loss_and_grads(Xtr, beta)
            te, _, _, _ = self.loss_and_grads(Xte, beta)
            hist["train"].append(tr)
            hist["test"].append(te)
            hist["rec"].append(rec)
            hist["kl"].append(kl)
        return hist

    def reconstruct(self, x):
        _, mu, _ = self.encode(x)
        return self.decode(mu)[1]

    def sample(self, n):
        return self.decode(self.rng.randn(n, self.z_dim))[1]


def show_grid(imgs, nrow, ncol, path, title=None):
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 0.9, nrow * 0.9))
    for a, im in zip(ax.ravel(), imgs):
        a.imshow(im.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        a.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    d = load_digits()
    X = d.data / 16.0
    y = d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("train", Xtr.shape, "test", Xte.shape)
    results = {}

    # 1. Main model: 2-D latent space (for visualisation)
    vae2 = VAE(z_dim=2)
    h2 = vae2.fit(Xtr, Xte, epochs=300)
    plt.figure(figsize=(6, 3.8))
    plt.plot(h2["train"], label="Train")
    plt.plot(h2["test"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (reconstruction + KL)")
    plt.title("VAE training curve (z=2)")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/loss_curve.png", dpi=160)
    plt.close()

    # 2. Latent space scatter
    _, mu, _ = vae2.encode(Xte)
    plt.figure(figsize=(5.6, 4.6))
    sc = plt.scatter(mu[:, 0], mu[:, 1], c=yte, cmap="tab10", s=14)
    plt.colorbar(sc, label="Digit")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("2-D latent space of test images")
    plt.tight_layout()
    plt.savefig(f"{OUT}/latent_space.png", dpi=160)
    plt.close()

    # 3. Latent manifold: decode a regular grid of z values
    n = 15
    grid = np.linspace(-2.5, 2.5, n)
    zs = np.array([[a, b] for b in grid[::-1] for a in grid])
    imgs = vae2.decode(zs)[1]
    tiles = np.pad(imgs.reshape(n * n, 8, 8), ((0, 0), (1, 1), (1, 1)))   # 1-px white gap per tile
    big = tiles.reshape(n, n, 10, 10).transpose(0, 2, 1, 3).reshape(n * 10, n * 10)
    plt.figure(figsize=(5.6, 5.6))
    plt.imshow(big, cmap="gray_r")
    plt.axis("off")
    plt.title("Generated digits across the latent space")
    plt.tight_layout()
    plt.savefig(f"{OUT}/latent_manifold.png", dpi=160)
    plt.close()

    # 4. Latent-dimension study (quantitative evaluation)
    dims = [2, 4, 8, 16]
    study, models = [], {}
    for zd in dims:
        m = VAE(z_dim=zd)
        hh = m.fit(Xtr, Xte, epochs=300)
        models[zd] = m
        mse = float(np.mean((m.reconstruct(Xte) - Xte) ** 2))
        study.append({"z_dim": zd, "test_loss": float(hh["test"][-1]), "rec": float(hh["rec"][-1]),
                      "kl": float(hh["kl"][-1]), "test_mse": mse})
        print(study[-1], flush=True)
    results["study"] = study

    # 5. Original vs reconstructed
    orig = Xte[:10]
    rows = np.vstack([orig] + [models[z].reconstruct(orig) for z in (2, 8, 16)])
    fig, ax = plt.subplots(4, 10, figsize=(9, 3.8))
    labs = ["Original", "z=2", "z=8", "z=16"]
    for r in range(4):
        for c in range(10):
            ax[r, c].imshow(rows[r * 10 + c].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
        ax[r, 0].set_ylabel(labs[r], rotation=0, labelpad=28, fontsize=9, va="center")
    plt.tight_layout()
    plt.savefig(f"{OUT}/reconstructions.png", dpi=160)
    plt.close()

    # 6. New samples drawn from N(0, I) using the z=16 model
    best = models[16]
    best.rng = np.random.RandomState(7)
    show_grid(best.sample(40), 4, 10, f"{OUT}/generated_samples.png",
              "New digits sampled from the prior (z=16)")

    # 7. Quality of generated samples: kNN-classifier confidence and nearest-neighbour distance
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(3).fit(Xtr, ytr)
    gen = best.sample(500)
    conf = knn.predict_proba(gen).max(1)
    dist = np.array([np.min(np.linalg.norm(Xtr - g, axis=1)) for g in gen])
    base = np.array([np.min(np.linalg.norm(Xtr - t, axis=1)) for t in Xte[:500]])
    results["generation"] = {"knn_confidence_mean": float(conf.mean()),
                             "share_conf_ge_0.67": float((conf >= 0.66).mean()),
                             "nn_dist_generated": float(dist.mean()),
                             "nn_dist_real_test": float(base.mean())}
    results["sizes"] = {"train": int(len(Xtr)), "test": int(len(Xte))}
    json.dump(results, open(f"{OUT}/metrics.json", "w"), indent=2)
    print(json.dumps(results, indent=2))

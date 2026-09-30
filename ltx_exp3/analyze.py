"""Experiment 3 analysis: bf16 vs fp32 Euler step at the same prompt/seed."""
import sys, numpy as np, torch, av

tag = sys.argv[1]  # e.g. 145f24
A = sys.argv[2] if len(sys.argv) > 2 else "bf16"
B = sys.argv[3] if len(sys.argv) > 3 else "fp32"
print(f"comparing {A} vs {B} @ {tag}")
d = "/home/rsalman/tt-metal/ltx_exp3"


def pcc(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_lat(p):
    t = torch.load(p, map_location="cpu")
    if isinstance(t, dict):
        t = next(v for v in t.values() if torch.is_tensor(v))
    return t.float().numpy()


la, lb = load_lat(f"{d}/audio_latent_{A}_{tag}.pt"), load_lat(f"{d}/audio_latent_{B}_{tag}.pt")
print(
    f"audio latent {la.shape}: PCC={pcc(la, lb):.5f}  rel diff={np.linalg.norm(la-lb)/np.linalg.norm(la):.4f}  max|d|={np.abs(la-lb).max():.3f}  std {la.std():.3f}/{lb.std():.3f}"
)


def read(p):
    c = av.open(p)
    frames = []
    audio = []
    for f in c.decode(video=0):
        frames.append(f.to_ndarray(format="gray"))
    c.seek(0)
    for f in c.decode(audio=0):
        audio.append(f.to_ndarray())
    c.close()
    return np.stack(frames), np.concatenate(audio, axis=-1)


va, aa = read(f"{d}/euler_{A}_{tag}_seed10.mp4")
vb, ab = read(f"{d}/euler_{B}_{tag}_seed10.mp4")
n = min(aa.shape[-1], ab.shape[-1])
aa, ab = aa[..., :n], ab[..., :n]
print(f"audio wave {aa.shape}: PCC={pcc(aa, ab):.5f}  rms {np.sqrt((aa**2).mean()):.4f}/{np.sqrt((ab**2).mean()):.4f}")
m = min(len(va), len(vb))
va, vb = va[:m].astype(np.float64), vb[:m].astype(np.float64)
mse = ((va - vb) ** 2).mean(axis=(1, 2))
psnr = 10 * np.log10(255**2 / np.maximum(mse, 1e-6))
print(
    f"video {va.shape}: mean PSNR={psnr.mean():.2f} dB  min={psnr.min():.2f}  frame0={psnr[0]:.2f}  last={psnr[-1]:.2f}  frame PCC={pcc(va, vb):.5f}"
)

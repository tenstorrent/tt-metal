"""Compare a reference dump (ref_driver.py) with a ttnn dump (LTX_DUMP_LATENT_DIR) stage by stage."""
import sys, os, torch

ref, tt = sys.argv[1], sys.argv[2]


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


print(f"{'tensor':<12}{'ref shape':<22}{'tt shape':<22}{'PCC':>8}{'rel err':>9}{'ref std':>9}{'tt std':>8}")
for name in ["embeds", "s1_video", "s1_audio", "upsampled", "s2_video", "s2_audio"]:
    pr, pt = os.path.join(ref, name + ".pt"), os.path.join(tt, name + ".pt")
    if not (os.path.exists(pr) and os.path.exists(pt)):
        print(f"{name:<12} missing ({'ref' if not os.path.exists(pr) else 'tt'})")
        continue
    a, b = torch.load(pr), torch.load(pt)
    det = (
        lambda t: t.detach()
        if torch.is_tensor(t)
        else {k: (v.detach() if torch.is_tensor(v) else v) for k, v in t.items()}
    )
    a, b = det(a), det(b)
    pairs = [(name, a, b)] if not isinstance(a, dict) else [(f"{name}/{k}", a[k], b[k]) for k in ("video", "audio")]
    for n, x, y in pairs:
        x, y = x.float(), y.float()
        if x.numel() != y.numel():
            print(f"{n:<12}{str(tuple(x.shape)):<22}{str(tuple(y.shape)):<22} numel mismatch")
            continue
        y = y.reshape(x.shape)
        print(
            f"{n:<12}{str(tuple(x.shape)):<22}{str(tuple(y.shape)):<22}{pcc(x, y):>8.4f}{rel(y, x):>9.4f}{x.std():>9.3f}{y.std():>8.3f}"
        )

# --- final media (optional): ref out.mp4 / audio.pt vs tt out.mp4 ---
try:
    import av, numpy as np

    def read(p):
        c = av.open(p)
        fr = []
        au = []
        for f in c.decode(video=0):
            fr.append(f.to_ndarray(format="gray"))
        c.seek(0)
        for f in c.decode(audio=0):
            au.append(f.to_ndarray())
        c.close()
        return np.stack(fr), np.concatenate(au, axis=-1)

    rp, tp = os.path.join(ref, "out.mp4"), os.path.join(tt, "out.mp4")
    if os.path.exists(rp) and os.path.exists(tp):
        va, aa = read(rp)
        vb, ab = read(tp)
        n = min(aa.shape[-1], ab.shape[-1])
        aa, ab = aa[..., :n].astype(np.float64), ab[..., :n].astype(np.float64)
        m = min(len(va), len(vb))
        va, vb = va[:m].astype(np.float64), vb[:m].astype(np.float64)
        mse = ((va - vb) ** 2).mean(axis=(1, 2))
        psnr = 10 * np.log10(255**2 / np.maximum(mse, 1e-6))
        print(
            f"final audio (mp4 vs mp4, {aa.shape}): PCC={pcc(torch.from_numpy(aa), torch.from_numpy(ab)):.4f}  rms {np.sqrt((aa**2).mean()):.4f}/{np.sqrt((ab**2).mean()):.4f}"
        )
        print(
            f"final video ({m} frames): mean PSNR={psnr.mean():.2f} dB  min={psnr.min():.2f}  frame PCC={pcc(torch.from_numpy(va), torch.from_numpy(vb)):.4f}"
        )
except ImportError:
    pass

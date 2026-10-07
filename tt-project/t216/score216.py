"""#216 scoring on the blx01 host (CPU): stride (2,4,4) vs stride 1 per seed, plus each arm vs #214's
unoptimized reference and the stride-1 seed-to-seed floor. Metrics on RGB (BT.709 limited range, the
mp4 export's decode) and luma. Seam ratio: mean |luma grad| across the stride grid's boundaries over
the mean at all other offsets (1.0 = no seam): 16 px in H/W (stride 4 sites x patch 4), 2 frames in T.
Usage: score216.py OUT_DIR REF_DIR"""
import json, sys
import numpy as np
from PIL import Image

W, H, T = 1920, 1088, 145
FR = W * H * 3 // 2
out, refdir = sys.argv[1], sys.argv[2]
A, B = "1x1x1", "2x4x4"


def load(p):
    return np.fromfile(p, np.uint8).reshape(T, FR)


def rgb(fr):
    y = fr[: W * H].reshape(H, W).astype(np.float32)
    u = fr[W * H : W * H * 5 // 4].reshape(H // 2, W // 2).repeat(2, 0).repeat(2, 1).astype(np.float32)
    v = fr[W * H * 5 // 4 :].reshape(H // 2, W // 2).repeat(2, 0).repeat(2, 1).astype(np.float32)
    y, u, v = (y - 16) * 1.164, u - 128, v - 128
    return np.clip(np.stack([y + 1.793 * v, y - 0.213 * u - 0.533 * v, y + 2.112 * u], -1), 0, 255)


def compare(x, z):
    se, n, sx, sz, sxx, szz, sxz, frames, yse = 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [], 0.0
    for t in range(T):
        a, b = rgb(x[t]).astype(np.float64), rgb(z[t]).astype(np.float64)
        d = ((a - b) ** 2).mean()
        se += d
        frames.append(10 * np.log10(255**2 / max(d, 1e-9)))
        yse += ((x[t, : W * H].astype(np.float64) - z[t, : W * H]) ** 2).mean()
        sx += a.sum()
        sz += b.sum()
        sxx += (a * a).sum()
        szz += (b * b).sum()
        sxz += (a * b).sum()
        n += a.size
    cov = sxz / n - sx / n * sz / n
    pcc = cov / np.sqrt((sxx / n - (sx / n) ** 2) * (szz / n - (sz / n) ** 2))
    return {
        "psnr_rgb": 10 * np.log10(255**2 / max(se / T, 1e-9)),
        "psnr_min_frame": min(frames),
        "worst_frame": int(np.argmin(frames)),
        "psnr_y": 10 * np.log10(255**2 / max(yse / T, 1e-9)),
        "pcc_rgb": pcc,
    }


def seam(x):
    g = x[:, : W * H].reshape(T, H, W).astype(np.float32)
    dw = np.abs(np.diff(g, axis=2)).mean(axis=(0, 1))
    dh = np.abs(np.diff(g, axis=1)).mean(axis=(0, 2))
    dt = np.abs(np.diff(g[1:], axis=0)).mean(axis=(1, 2))  # frame 0 is the causal first frame

    def ratio(d, p):
        on = (np.arange(d.shape[0]) % p) == p - 1
        return float(d[on].mean() / d[~on].mean())

    return {"w": ratio(dw, 16), "h": ratio(dh, 16), "t": ratio(dt, 2)}


rep = {"per_seed": {}, "seam": {A: [], B: []}, "stills": []}
for s in range(5):
    xa, xb = load(f"{out}/s{A}_seed{s}.yuv"), load(f"{out}/s{B}_seed{s}.yuv")
    ref = load(f"{refdir}/ref_dvx_seed{s}.yuv")
    r = {"B_vs_A": compare(xb, xa), "A_vs_ref214": compare(xa, ref), "B_vs_ref214": compare(xb, ref)}
    if s < 4:
        r["floor_A_seed_vs_next"] = compare(xa, load(f"{out}/s{A}_seed{s + 1}.yuv"))
    rep["seam"][A].append(seam(xa))
    rep["seam"][B].append(seam(xb))
    rep["per_seed"][s] = r
    t = 72
    ia, ib = rgb(xa[t]), rgb(xb[t])
    Image.fromarray(ia.astype(np.uint8)).save(f"{out}/still_s{A}_seed{s}_f{t}.png")
    Image.fromarray(ib.astype(np.uint8)).save(f"{out}/still_s{B}_seed{s}_f{t}.png")
    Image.fromarray(np.clip(np.abs(ib - ia) * 8, 0, 255).astype(np.uint8)).save(f"{out}/diff8x_seed{s}_f{t}.png")
    # 1:1 crop at the image centre, stride 1 left / stride 2x4x4 right, to look for 16 px seams
    cy, cx = H // 2 - 256, W // 2 - 256
    Image.fromarray(
        np.concatenate([ia[cy : cy + 512, cx : cx + 512], ib[cy : cy + 512, cx : cx + 512]], 1).astype(np.uint8)
    ).save(f"{out}/crop512_AB_seed{s}_f{t}.png")
    rep["stills"].append(f"{out}/still_s{B}_seed{s}_f{t}.png")
    b = r["B_vs_A"]
    print(
        f"seed {s}: 2x4x4 vs 1: PSNR {b['psnr_rgb']:.2f} dB (min frame {b['psnr_min_frame']:.2f} @ {b['worst_frame']}), "
        f"Y {b['psnr_y']:.2f}, PCC {b['pcc_rgb']:.6f} | A vs #214 ref {r['A_vs_ref214']['psnr_rgb']:.2f} dB "
        f"PCC {r['A_vs_ref214']['pcc_rgb']:.6f} | B vs ref {r['B_vs_ref214']['psnr_rgb']:.2f} dB"
        + (f" | floor {r['floor_A_seed_vs_next']['psnr_rgb']:.2f} dB" if s < 4 else ""),
        flush=True,
    )
for arm in (A, B):
    v = rep["seam"][arm]
    print(
        f"seam ratio {arm}@16px/2f: w {np.mean([x['w'] for x in v]):.3f} h {np.mean([x['h'] for x in v]):.3f} t {np.mean([x['t'] for x in v]):.3f}"
    )
json.dump(rep, open(f"{out}/scores.json", "w"), indent=1, default=float)
print("SCORE216_DONE")

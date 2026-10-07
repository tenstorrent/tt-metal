"""t225 post (blx01, CPU): per seed, PCC/PSNR of 2d vs 1d and of each arm vs the #214 host-noise reference
(ref_dvx_seed{N}.yuv). Device noise differs between arms, so these numbers only size the noise effect; the verdict is
visual + VBench. Also writes still_seed{N}_f{F}.jpg (top 1d, bottom 2d, half size) and crop_seed{N}_f{F}.png
(512x512 centre crops at full res: 1d | 2d | 8x abs diff).
yuv420p 1920x1088, 145 frames. Usage: post.py <out_dir> <ref_dir>"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

W, H, T = 1920, 1088, 145
FR = W * H * 3 // 2
out, ref = Path(sys.argv[1]), Path(sys.argv[2])


def load(p):
    a = np.fromfile(p, np.uint8)
    assert a.size == FR * T, (p, a.size)
    return a.reshape(T, FR)


def stats(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    mse_f = ((a - b) ** 2).mean(1)
    psnr = lambda m: float(10 * np.log10(255**2 / max(m, 1e-12)))
    w = int(mse_f.argmax())
    return dict(
        pcc=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
        psnr=psnr(mse_f.mean()),
        worst_frame=w,
        worst_psnr=psnr(mse_f[w]),
    )


def rgb(frame):
    Y = frame[: W * H].reshape(H, W)
    U = frame[W * H : W * H * 5 // 4].reshape(H // 2, W // 2).repeat(2, 0).repeat(2, 1)
    V = frame[W * H * 5 // 4 :].reshape(H // 2, W // 2).repeat(2, 0).repeat(2, 1)
    return np.asarray(Image.fromarray(np.stack([Y, U, V], -1), "YCbCr").convert("RGB"))


res = {}
for s in range(5):
    a, b, r = load(out / f"1d_seed{s}.yuv"), load(out / f"2d_seed{s}.yuv"), load(ref / f"ref_dvx_seed{s}.yuv")
    res[s] = {"2d_vs_1d": stats(b, a), "1d_vs_ref": stats(a, r), "2d_vs_ref": stats(b, r)}
    print(s, json.dumps(res[s]), flush=True)
    for f in (72,) if s else (0, 72, 144):
        A, B = rgb(a[f]), rgb(b[f])
        Image.fromarray(np.concatenate([A, B], 0)).resize((W // 2, H)).save(out / f"still_seed{s}_f{f}.jpg", quality=90)
        y0, x0 = H // 2 - 256, W // 2 - 256
        ca, cb = A[y0 : y0 + 512, x0 : x0 + 512], B[y0 : y0 + 512, x0 : x0 + 512]
        d = np.clip(np.abs(ca.astype(np.int16) - cb.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
        Image.fromarray(np.concatenate([ca, cb, d], 1)).save(out / f"crop_seed{s}_f{f}.png")
(out / "cmp.json").write_text(json.dumps(res, indent=1))
for k in ("2d_vs_1d", "1d_vs_ref", "2d_vs_ref"):
    print(f"T225_CMP {k} psnr " + " ".join(f"{res[s][k]['psnr']:.2f}" for s in range(5)), flush=True)

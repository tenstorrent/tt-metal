"""PCC/PSNR of each fidelity arm's yuv420p decode against the default (HiFi4) arm, per seed.

Usage: cmp335.py <out_dir> <ref_arm> <arm> [<arm>...]. Reads <out_dir>/<arm>/seed{N}.yuv; prints one CMP line per
arm and seed and writes <out_dir>/cmp_<arm>.json. PSNR is over all yuv bytes (peak 255) and over Y alone.
"""

import json
import sys
from pathlib import Path

import numpy as np

T, H, W = 145, 1088, 1920
FRAME = H * W * 3 // 2
out, ref_arm, arms = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


for arm in arms:
    rows = {}
    for p in sorted((out / ref_arm).glob("seed*.yuv")):
        q = out / arm / p.name
        if not q.exists():
            continue
        a = np.fromfile(p, np.uint8).reshape(T, FRAME).astype(np.float32)
        b = np.fromfile(q, np.uint8).reshape(T, FRAME).astype(np.float32)
        ya, yb = a[:, : H * W], b[:, : H * W]
        pcc = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        per_frame = [psnr(a[t], b[t]) for t in range(T)]
        rows[p.stem] = {
            "identical": bool((a == b).all()),
            "pcc": pcc,
            "psnr": psnr(a, b),
            "psnr_y": psnr(ya, yb),
            "psnr_min_frame": min(per_frame),
            "max_abs": float(np.abs(a - b).max()),
        }
        r = rows[p.stem]
        print(
            f"CMP {arm} vs {ref_arm} {p.stem}: identical={r['identical']} pcc={pcc:.6f} psnr={r['psnr']:.2f} "
            f"psnr_y={r['psnr_y']:.2f} psnr_min_frame={r['psnr_min_frame']:.2f} max_abs={r['max_abs']:.0f}",
            flush=True,
        )
    (out / f"cmp_{arm}.json").write_text(json.dumps(rows, indent=1))

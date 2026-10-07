"""t212 score: port yuv (score/out) vs #214 reference yuv (diffvae/ref), per seed and per frame.
yuv420p 1920x1088, 145 frames. PCC over all bytes; PSNR on Y and on all planes (peak 255)."""

import json
import sys
from pathlib import Path

import numpy as np

W, H, T = 1920, 1088, 145
FR = W * H * 3 // 2
ref_dir, out_dir, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
res = {}
for s in range(5):
    a = np.fromfile(ref_dir / f"ref_dvx_seed{s}.yuv", np.uint8)
    b = np.fromfile(out_dir / f"ref_dvx_seed{s}.yuv", np.uint8)
    assert a.size == b.size == FR * T, (s, a.size, b.size)
    a, b = a.reshape(T, FR).astype(np.float32), b.reshape(T, FR).astype(np.float32)
    mse_f = ((a - b) ** 2).mean(1)
    mse_y = ((a[:, : W * H] - b[:, : W * H]) ** 2).mean()
    psnr = lambda m: float(10 * np.log10(255**2 / max(m, 1e-12)))
    pcc = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    worst = int(mse_f.argmax())
    res[s] = dict(
        pcc=pcc,
        psnr=psnr(mse_f.mean()),
        psnr_y=psnr(mse_y),
        worst_frame=worst,
        worst_psnr=psnr(mse_f[worst]),
        identical=bool((a == b).all()),
    )
    print(s, res[s], flush=True)
times = json.loads((out_dir / "decode_times.json").read_text())
res["decode_s"] = times
res["mean_decode_s"] = sum(times.values()) / len(times)
dst.write_text(json.dumps(res, indent=1))
print("T212S_CMP", json.dumps({k: v for k, v in res.items() if k in ("mean_decode_s",)}))

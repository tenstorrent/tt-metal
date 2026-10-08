# t221: host-side sanity check of bf16 vs bf8 LTX-2.3 output (same seed): PCC/PSNR over decoded mp4 frames, stills.
import sys
import av
import numpy as np
from PIL import Image


def frames(path):
    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)]).astype(np.float64)


a, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
fa, fb = frames(a), frames(b)
n = min(len(fa), len(fb))
fa, fb = fa[:n], fb[:n]
pcc = np.corrcoef(fa.ravel(), fb.ravel())[0, 1]
mse = np.mean((fa - fb) ** 2)
psnr = 10 * np.log10(255.0**2 / mse)
per = [10 * np.log10(255.0**2 / max(np.mean((fa[i] - fb[i]) ** 2), 1e-9)) for i in range(n)]
print(
    f"frames={n} shape={fa.shape[1:]} PCC={pcc:.5f} PSNR={psnr:.2f} dB per-frame PSNR min={min(per):.2f} max={max(per):.2f}"
)
k = n // 2
Image.fromarray(fb[k].astype(np.uint8)).save(f"{out}/still_bf16_seed0_f{k}.png")
Image.fromarray(np.concatenate([fa[k], fb[k]], axis=1).astype(np.uint8)).save(f"{out}/still_bf8_vs_bf16_seed0_f{k}.png")

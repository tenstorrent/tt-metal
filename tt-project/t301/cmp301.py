# t301: main vs PR arm, same seed/prompt: PCC/PSNR over decoded mp4 frames (both are lossy encodes; the PR's
# export is x264 ultrafast crf 20 vs main's veryfast crf 23, so the mp4 md5 always differs) and a side-by-side still.
import sys
import av
import numpy as np
from PIL import Image


def frames(path):
    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)]).astype(np.float64)


a, b, out, tag = sys.argv[1:5]
fa, fb = frames(a), frames(b)
n = min(len(fa), len(fb))
fa, fb = fa[:n], fb[:n]
pcc = np.corrcoef(fa.ravel(), fb.ravel())[0, 1]
mse = np.mean((fa - fb) ** 2)
psnr = 10 * np.log10(255.0**2 / max(mse, 1e-12))
per = [10 * np.log10(255.0**2 / max(np.mean((fa[i] - fb[i]) ** 2), 1e-9)) for i in range(n)]
print(
    f"{tag} frames={n}/{len(fa)} shape={fa.shape[1:]} PCC={pcc:.6f} PSNR={psnr:.2f} dB per-frame min={min(per):.2f} max={max(per):.2f}"
)
k = n // 2
Image.fromarray(np.concatenate([fa[k], fb[k]], axis=1).astype(np.uint8)).save(f"{out}/still_main_vs_pr_{tag}_f{k}.png")

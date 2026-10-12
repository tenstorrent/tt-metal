# t452: PSNR of an arm's saved video (numpy uint8 (frames, 1.5*H, W) yuv420, Y in the first H rows) vs the dense
# arm's: Y plane, whole frame, per-frame min/median; plus latent PCC. Args: <dense out dir> <arm out dir>. CPU only.
import sys
from pathlib import Path

import numpy as np
import torch


def load(d, kind):
    return torch.load(next(Path(d).glob(f"*_{kind}*.pt")), map_location="cpu", weights_only=False)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


ref, arm = sys.argv[1], sys.argv[2]
va, vb = load(ref, "video").astype(np.float32), load(arm, "video").astype(np.float32)
h = va.shape[1] * 2 // 3
per = [psnr(va[i, :h], vb[i, :h]) for i in range(va.shape[0])]
print(
    f"{Path(arm).name} vs {Path(ref).name}: Y PSNR {psnr(va[:, :h], vb[:, :h]):.2f} dB, yuv PSNR {psnr(va, vb):.2f} dB, "
    f"Y per-frame min {min(per):.2f} median {float(np.median(per)):.2f} (frame {int(np.argmin(per))} worst)"
)
la, lb = load(ref, "latents").flatten().double(), load(arm, "latents").flatten().double()
print(f"latent PCC {torch.corrcoef(torch.stack([la, lb]))[0, 1].item():.4f}")

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Score test_vae_ltx_conv_precision_ab.py outputs on CPU against the base arm.

Per arm: per-frame PSNR over all yuv420 planes (min/mean), mean SSIM on Y, and a PNG still of the middle
frame (plus a 4x crop and an amplified diff vs base). Writes summary.json next to the yuv files.
Usage: python compare_conv_precision_ab.py <AB_OUT_DIR> [ref_arm]
"""

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _psnr_per_frame(a, b):
    mse = ((a.float() - b.float()) ** 2).flatten(1).mean(1)
    return [float("inf") if m == 0 else 10 * math.log10(255.0**2 / m) for m in mse.tolist()]


def _ssim_y(a, b):
    """Mean SSIM per frame on Y (11-tap Gaussian, sigma 1.5, standard constants)."""
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    k = torch.outer(g, g)

    def blur(x):
        return F.conv2d(x, k.view(1, 1, 11, 11))

    out = []
    for fa, fb in zip(a.float(), b.float()):
        x, y = fa[None, None], fb[None, None]
        mx, my = blur(x), blur(y)
        sxx, syy, sxy = blur(x * x) - mx**2, blur(y * y) - my**2, blur(x * y) - mx * my
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
        out.append(float(s.mean()))
    return out


def _rgb(frame, h, w):
    """yuv420p (h*3/2, w) uint8 -> (h, w, 3) uint8, BT.709 limited range."""
    y = frame[:h].float()
    u = frame[h : h + h // 4].reshape(h // 2, w // 2).float()
    v = frame[h + h // 4 :].reshape(h // 2, w // 2).float()
    u = u.repeat_interleave(2, 0).repeat_interleave(2, 1) - 128
    v = v.repeat_interleave(2, 0).repeat_interleave(2, 1) - 128
    yy = (y - 16) * 255 / 219
    u, v = u * 255 / 224, v * 255 / 224
    r = yy + 1.5748 * v
    g = yy - 0.1873 * u - 0.4681 * v
    b = yy + 1.8556 * u
    return torch.stack([r, g, b], -1).clamp(0, 255).round().byte().numpy()


def main(out_dir, ref_arm="base"):
    ref = torch.load(os.path.join(out_dir, f"yuv_{ref_arm}.pt"))
    t, h32, w = ref.shape
    h = h32 * 2 // 3
    mid = t // 2
    summary = {}
    arms = sorted(f[4:-3] for f in os.listdir(out_dir) if f.startswith("yuv_") and f.endswith(".pt"))
    for arm in arms:
        out = ref if arm == ref_arm else torch.load(os.path.join(out_dir, f"yuv_{arm}.pt"))
        psnr = _psnr_per_frame(out, ref)
        ssim = _ssim_y(out[:, :h], ref[:, :h])
        finite = [p for p in psnr if math.isfinite(p)]
        summary[arm] = {
            "psnr_min": min(psnr),
            "psnr_mean": sum(finite) / len(finite) if finite else float("inf"),
            "psnr_min_frame": int(np.argmin(psnr)),
            "ssim_y_min": min(ssim),
            "ssim_y_mean": sum(ssim) / len(ssim),
            "max_abs_diff": int((out.int() - ref.int()).abs().max()),
        }
        rgb = _rgb(out[mid], h, w)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"still_{arm}.png"))
        ch, cw = h // 4, w // 4
        crop = rgb[h // 2 - ch // 2 : h // 2 + ch // 2, w // 2 - cw // 2 : w // 2 + cw // 2]
        Image.fromarray(crop).resize((cw * 4, ch * 4), Image.NEAREST).save(os.path.join(out_dir, f"crop_{arm}.png"))
        if arm != ref_arm:
            diff = (out[mid, :h].int() - ref[mid, :h].int()).abs() * 16
            Image.fromarray(diff.clamp(0, 255).byte().numpy()).save(os.path.join(out_dir, f"diff16_{arm}.png"))
        print(arm, json.dumps(summary[arm]))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])

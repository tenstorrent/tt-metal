"""t460 quality: integrated (device 1080p) vs t397 (768p) per saved run, CPU only.

Latents: bit-identity and max abs diff. Video: the base yuv420 planes are bicubic-resized on the host
(torch, align_corners=False, the device upscale's reference) to the int arm's size, then PSNR (uint8
peak) and PCC per plane set, over all frames. Also writes a side-by-side still (frame 79) per seed.
Usage: python -I q460.py <t460 dir>
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

root = Path(sys.argv[1])
torch.set_num_threads(16)


def planes(yuv, h, w):
    y = yuv[:, :h]
    uv = yuv[:, h:].reshape(yuv.shape[0], 2, h // 2, w // 2)
    return y, uv[:, 0], uv[:, 1]


def resize(p, size):
    t = torch.from_numpy(p).float().unsqueeze(1)
    return F.interpolate(t, size=size, mode="bicubic", align_corners=False).squeeze(1).round().clamp(0, 255)


def stats(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    mse = ((a - b) ** 2).mean().item()
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return psnr, pcc


def rgb(y, u, v):
    y, u, v = (x.float() for x in (y, u, v))
    u = F.interpolate(u[None, None], size=y.shape, mode="nearest")[0, 0] - 128
    v = F.interpolate(v[None, None], size=y.shape, mode="nearest")[0, 0] - 128
    y = (y - 16) * 1.164
    return torch.stack([y + 1.596 * v, y - 0.392 * u - 0.813 * v, y + 2.017 * u], -1).clamp(0, 255).byte().numpy()


rows = []
for base in sorted(root.glob("out_b*")):
    tag = base.name[5:]
    intd = root / f"out_i{tag}"
    lat_b, lat_i = (next(d.glob("*_latents.pt")) for d in (base, intd))
    lb, li = torch.load(lat_b), torch.load(lat_i)
    bit = torch.equal(lb, li)
    lmax = (lb - li).abs().max().item()
    vb = torch.load(next(base.glob("*_video_yuv420.pt")), weights_only=False)
    vi = torch.load(next(intd.glob("*_video_yuv420.pt")), weights_only=False)
    hb, wb = vb.shape[1] * 2 // 3, vb.shape[2]
    hi, wi = vi.shape[1] * 2 // 3, vi.shape[2]
    pb, pi = planes(vb, hb, wb), planes(vi, hi, wi)
    ys, uvs = [], []
    up_y, up_u, up_v = [], [], []
    for s in range(0, vb.shape[0], 16):
        y = resize(pb[0][s : s + 16], (hi, wi))
        u = resize(pb[1][s : s + 16], (hi // 2, wi // 2))
        v = resize(pb[2][s : s + 16], (hi // 2, wi // 2))
        up_y.append(y), up_u.append(u), up_v.append(v)
    y, u, v = torch.cat(up_y), torch.cat(up_u), torch.cat(up_v)
    iy, iu, iv = (torch.from_numpy(np.ascontiguousarray(p)).float() for p in pi)
    psnr_y, pcc_y = stats(y, iy)
    psnr_all, pcc_all = stats(
        torch.cat([y.flatten(), u.flatten(), v.flatten()]), torch.cat([iy.flatten(), iu.flatten(), iv.flatten()])
    )
    f = min(79, vb.shape[0] - 1)
    still = np.concatenate([rgb(y[f], u[f], v[f]), rgb(iy[f], iu[f], iv[f])], axis=1)
    try:
        from PIL import Image

        Image.fromarray(still).save(root / f"still_{tag}_f{f}_base-bicubic_vs_int.png")
    except ImportError:
        np.save(root / f"still_{tag}_f{f}.npy", still)
    rows.append((tag, bit, lmax, f"{vb.shape}->{vi.shape}", psnr_y, pcc_y, psnr_all, pcc_all))
    print(
        f"{tag}: latents bit-identical={bit} max|d|={lmax:.3g}; video {hb}x{wb}->{hi}x{wi}; "
        f"Y PSNR {psnr_y:.2f} dB PCC {pcc_y:.5f}; YUV PSNR {psnr_all:.2f} dB PCC {pcc_all:.5f}",
        flush=True,
    )
print("T460Q_DONE")

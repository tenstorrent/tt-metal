# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Required reference check for the conv VAE decode A/B harnesses.

Arms that only match each other can share one bug (a wrong blocking table, a dropped halo), so every arm's
yuv420p decode is also compared against a stored reference decode of the same latent: overall PSNR over all
planar bytes, SSIM-Y, and Y PSNR in bands around the chip-boundary columns and rows of the mesh split.
Fails under PSNR_MIN_DB overall or SEAM_PSNR_MIN_DB at either seam set.

Reference: $LTX_VAE_REF, else $LTX_VAE_REF_DIR (default /var/tmp/fasth3/vae_ref) /
ltx_conv_<H>x<W>x<F>_<latent md5[:12]>.pt. Accepted contents: a yuv420p planar uint8 tensor (T, H*3//2, W)
as the harnesses save it, or a dict with "yuv" (that tensor) or "video" (float BCTHW in [-1, 1], e.g. a
DiffVAE decode, converted here with BT.601 limited range), plus an optional "latent_md5" that must match.
LTX_VAE_REF_RECORD=1 writes a missing reference from the current arm, only when that arm runs without the
halo path (LTX_VAE_HALO_ONLY=0).

Offline: python -m models.tt_dit.tests.models.ltx.tools.vae_ref_check --ref ref.pt --arm yuv_x.pt \
    --height 544 --width 960 --mesh 2,4 [--exact-shard]
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

PSNR_MIN_DB = 40.0
SEAM_PSNR_MIN_DB = 35.0
# A wrong halo at the latent-resolution convs spreads over tens of output pixels; 16 px each side catches
# it while keeping the band a small fraction of the frame.
SEAM_BAND_PX = 16
# LTX-2 decoder: three 2x spatial upsamples (compress_all, compress_all, compress_space), then a 4x unpatch.
LTX_SPATIAL_UPSAMPLES = (2, 2, 2)
LTX_PATCH = 4
REF_DIR_DEFAULT = "/var/tmp/fasth3/vae_ref"


def chip_boundaries(
    out_size: int,
    factor: int,
    *,
    upsamples=LTX_SPATIAL_UPSAMPLES,
    patch: int = LTX_PATCH,
    exact_shard: bool = False,
) -> list[int]:
    """Output-pixel positions of every chip boundary any decoder stage sees along one axis.

    The latent is padded up to a multiple of ``factor`` and split evenly, so the padded split's boundaries
    hold through every upsample; with exact_shard the split is redone on the logical extent after each
    upsample that makes it divide, which moves the boundaries. Boundaries inside the cropped-off padding
    are dropped.
    """
    total = patch * math.prod(upsamples)
    assert out_size % total == 0, f"{out_size} is not a multiple of the decoder's {total}x upscale"
    logical = out_size // total
    chip = -(-logical // factor)
    scale = total
    bounds = set()

    def add():
        bounds.update(p for p in (k * chip * scale for k in range(1, factor)) if p < out_size)

    add()
    for up in upsamples:
        logical, chip, scale = logical * up, chip * up, scale // up
        if exact_shard and logical % factor == 0:
            chip = logical // factor
        add()
    return sorted(bounds)


def seam_index(out_size: int, boundaries, band: int = SEAM_BAND_PX) -> np.ndarray:
    """Sorted unique indices within ``band`` of any boundary: [b - band, b + band)."""
    idx = set()
    for b in boundaries:
        idx.update(range(max(0, b - band), min(out_size, b + band)))
    return np.array(sorted(idx), dtype=np.int64)


def split_yuv420p(planar, height: int, width: int):
    """(T, H*3//2, W) or (T, H*W*3//2) uint8 -> (Y (T, H, W), chroma (T, H*W//2))."""
    t = torch.as_tensor(np.asarray(planar)).reshape(-1, height * width * 3 // 2)
    return t[:, : height * width].reshape(-1, height, width), t[:, height * width :]


def float_to_yuv420p(video_BCTHW: torch.Tensor) -> torch.Tensor:
    """Float RGB in [-1, 1] -> yuv420p planar uint8 (T, H*3//2, W), BT.601 limited range, 2x2 chroma mean."""
    x = video_BCTHW[0].float().clamp(-1, 1).add(1).mul(0.5).permute(1, 0, 2, 3)  # (T, 3, H, W) in [0, 1]
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    kr, kb = 0.299, 0.114
    kg = 1 - kr - kb
    y = 16 + 219 * (kr * r + kg * g + kb * b)
    cb = 128 + 224 * (b - (kr * r + kg * g + kb * b)) / (2 * (1 - kb))
    cr = 128 + 224 * (r - (kr * r + kg * g + kb * b)) / (2 * (1 - kr))
    cb, cr = (F.avg_pool2d(c[:, None], 2)[:, 0] for c in (cb, cr))
    t, h, w = y.shape
    planes = [p.round().clamp(0, 255).to(torch.uint8).reshape(t, -1) for p in (y, cb, cr)]
    return torch.cat(planes, dim=1).reshape(t, h * 3 // 2, w)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.double() - b.double()).pow(2).mean().item()
    return 99.0 if mse == 0 else 10 * math.log10(255.0**2 / mse)


def ssim_y(a: torch.Tensor, b: torch.Tensor, win: int = 7) -> torch.Tensor:
    """Per-frame mean SSIM of (T, H, W) uint8 luma, uniform window (skimage defaults)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    out = []
    for i in range(a.shape[0]):
        x, y = a[i, None, None].double(), b[i, None, None].double()
        mx, my = F.avg_pool2d(x, win, 1), F.avg_pool2d(y, win, 1)
        n = win * win
        cov = n / (n - 1)
        vx = (F.avg_pool2d(x * x, win, 1) - mx * mx) * cov
        vy = (F.avg_pool2d(y * y, win, 1) - my * my) * cov
        vxy = (F.avg_pool2d(x * y, win, 1) - mx * my) * cov
        s = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
        out.append(s.mean().item())
    return torch.tensor(out)


def compare(
    arm,
    ref,
    *,
    height: int,
    width: int,
    mesh_shape: tuple[int, int],
    exact_shard: bool = False,
    band: int = SEAM_BAND_PX,
) -> dict:
    """Metrics of one arm's yuv420p decode against the reference. mesh_shape is (H factor, W factor)."""
    arm_y, arm_c = split_yuv420p(arm, height, width)
    ref_y, ref_c = split_yuv420p(ref, height, width)
    if arm_y.shape != ref_y.shape:
        raise ValueError(f"frame count differs: arm {tuple(arm_y.shape)} vs ref {tuple(ref_y.shape)}")
    rows = chip_boundaries(height, mesh_shape[0], exact_shard=exact_shard)
    cols = chip_boundaries(width, mesh_shape[1], exact_shard=exact_shard)
    ri, ci = seam_index(height, rows, band), seam_index(width, cols, band)
    ssim = ssim_y(arm_y, ref_y)
    mse_all = (
        torch.cat(
            [(arm_y.double() - ref_y.double()).pow(2).flatten(), (arm_c.double() - ref_c.double()).pow(2).flatten()]
        )
        .mean()
        .item()
    )
    frame_psnr = [psnr(arm_y[i], ref_y[i]) for i in range(arm_y.shape[0])]
    return {
        "psnr": 99.0 if mse_all == 0 else 10 * math.log10(255.0**2 / mse_all),
        "psnr_y_frame_min": min(frame_psnr),
        "psnr_c": psnr(arm_c, ref_c),
        "ssim_y": ssim.mean().item(),
        "ssim_y_min": ssim.min().item(),
        "seam_col_psnr": psnr(arm_y[:, :, ci], ref_y[:, :, ci]) if len(ci) else 99.0,
        "seam_row_psnr": psnr(arm_y[:, ri, :], ref_y[:, ri, :]) if len(ri) else 99.0,
        "seam_cols": cols,
        "seam_rows": rows,
    }


def failures(metrics: dict, psnr_min: float = PSNR_MIN_DB, seam_min: float = SEAM_PSNR_MIN_DB) -> list[str]:
    out = []
    if metrics["psnr"] < psnr_min:
        out.append(f"psnr {metrics['psnr']:.2f} < {psnr_min}")
    for key in ("seam_col_psnr", "seam_row_psnr"):
        if metrics[key] < seam_min:
            out.append(f"{key} {metrics[key]:.2f} < {seam_min}")
    return out


def report_line(arm: str, metrics: dict, fails: list[str]) -> str:
    return (
        f"VAE_REF arm={arm} {'OK' if not fails else 'FAIL'} psnr={metrics['psnr']:.2f}"
        f" psnr_y_frame_min={metrics['psnr_y_frame_min']:.2f} psnr_c={metrics['psnr_c']:.2f}"
        f" ssim_y={metrics['ssim_y']:.5f} ssim_y_min={metrics['ssim_y_min']:.5f}"
        f" seam_col_psnr={metrics['seam_col_psnr']:.2f} seam_row_psnr={metrics['seam_row_psnr']:.2f}"
        f" seam_cols={metrics['seam_cols']} seam_rows={metrics['seam_rows']}"
        + (f" fail=[{'; '.join(fails)}]" if fails else "")
    )


def latent_md5(latent: torch.Tensor) -> str:
    return hashlib.md5(latent.float().contiguous().numpy().tobytes()).hexdigest()


def reference_path(height: int, width: int, num_frames: int, md5: str) -> str:
    path = os.environ.get("LTX_VAE_REF")
    if path:
        return path
    ref_dir = os.environ.get("LTX_VAE_REF_DIR", REF_DIR_DEFAULT)
    return os.path.join(ref_dir, f"ltx_conv_{height}x{width}x{num_frames}_{md5[:12]}.pt")


def load_reference(path: str, md5: str | None = None) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if md5 is not None and obj.get("latent_md5") not in (None, md5):
            raise ValueError(f"reference {path} was decoded from latent {obj['latent_md5']}, this run uses {md5}")
        obj = obj["yuv"] if "yuv" in obj else float_to_yuv420p(obj["video"])
    return torch.as_tensor(obj)


def check_against_reference(
    out,
    arm: str,
    *,
    latent: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    mesh_shape: tuple[int, int],
    exact_shard: bool = False,
) -> dict:
    """Harness entry point: compare ``out`` to the stored reference, print one VAE_REF line, assert."""
    md5 = latent_md5(latent)
    path = reference_path(height, width, num_frames, md5)
    out = torch.as_tensor(np.asarray(out)).clone()
    if not os.path.exists(path):
        if os.environ.get("LTX_VAE_REF_RECORD") != "1":
            raise AssertionError(
                f"no VAE reference at {path}: set LTX_VAE_REF, or LTX_VAE_REF_RECORD=1 on a halo-off arm"
            )
        if os.environ.get("LTX_VAE_HALO_ONLY") != "0":
            raise AssertionError(
                "LTX_VAE_REF_RECORD=1 needs LTX_VAE_HALO_ONLY=0: the reference must not use the halo path"
            )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"yuv": out, "latent_md5": md5, "source": f"arm={arm} LTX_VAE_HALO_ONLY=0"}, path)
        print(f"VAE_REF recorded arm={arm} path={path}")
    ref = load_reference(path, md5)
    metrics = compare(out, ref, height=height, width=width, mesh_shape=mesh_shape, exact_shard=exact_shard)
    fails = failures(metrics)
    print(report_line(arm, metrics, fails) + f" ref={path}", flush=True)
    assert not fails, f"arm {arm} vs reference {path}: {'; '.join(fails)}"
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arm", required=True, nargs="+", help="yuv .pt files saved by an A/B harness")
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--mesh", default="2,4", help="H factor,W factor of the run that produced the arms")
    ap.add_argument("--exact-shard", action="store_true")
    ap.add_argument("--band", type=int, default=SEAM_BAND_PX)
    args = ap.parse_args(argv)
    mesh = tuple(int(v) for v in args.mesh.split(","))
    ref = load_reference(args.ref)
    bad = 0
    for path in args.arm:
        m = compare(
            torch.load(path, map_location="cpu"),
            ref,
            height=args.height,
            width=args.width,
            mesh_shape=mesh,
            exact_shard=args.exact_shard,
            band=args.band,
        )
        fails = failures(m)
        bad += bool(fails)
        print(report_line(os.path.basename(path), m, fails))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

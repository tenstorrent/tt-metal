#!/usr/bin/env python3
"""DUSt3R/MASt3R on-device demo.

Runs `dust3r_forward` on a pair of CO3Dv2 frames and writes a side-by-side
visualisation to `media/` showing the input images, port-vs-reference depth
maps, and per-pixel confidence. Defaults to the same `apple` sequence used by
`eval_mast3r.py`.

Usage:
    python3 demo.py                                   # default pair @ 512×512
    python3 demo.py --img-h 256 --img-w 512           # non-square aspect
    python3 demo.py --frame-i 0 --frame-j 80          # different baseline
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Match test_mast3r.py / eval_mast3r.py import shim.
_TT_METAL_ROOT = "/home/ttuser/experiments/medgemma/tt-metal"
if _TT_METAL_ROOT not in sys.path:
    sys.path.insert(0, _TT_METAL_ROOT)
    sys.path.insert(1, os.path.join(_TT_METAL_ROOT, "ttnn"))
    sys.path.insert(2, os.path.join(_TT_METAL_ROOT, "tools"))
os.chdir(_TT_METAL_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

_MAST3R_ROOT = "/home/ttuser/experiments/mast3r/tt-metal/models/demos/mast3r"
sys.path.insert(0, _MAST3R_ROOT)

from reference.torch_dust3r import load_checkpoint, load_dust3r  # noqa: E402

# Reuse the eval harness's preprocessing + activations.
sys.path.insert(0, "/home/ttuser/experiments/mast3r")
from eval_mast3r import (  # noqa: E402
    load_image_for_dust3r, activate_pts3d, activate_conf, pcc,
)


def colormap_depth(z: np.ndarray) -> np.ndarray:
    """Turbo-like colormap on inverse-depth percentile range. (H, W) → (H, W, 3) uint8."""
    valid = np.isfinite(z) & (z > 1e-6)
    if not valid.any():
        return np.zeros((*z.shape, 3), dtype=np.uint8)
    inv = np.where(valid, 1.0 / np.clip(z, 1e-6, None), 0.0)
    lo, hi = np.percentile(inv[valid], (2, 98))
    inv = np.clip((inv - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    # Simple turbo polynomial (matplotlib-free).
    r = np.clip(34.61 + inv * (1172.33 + inv * (-10793.56 + inv * (33300.12 + inv * (-38394.49 + inv * 14825.05)))), 0, 255)
    g = np.clip(23.31 + inv * (557.33 + inv * (1225.33 + inv * (-3574.96 + inv * (1073.77 + inv * 707.56)))), 0, 255)
    b = np.clip(27.2 + inv * (3211.1 + inv * (-15327.97 + inv * (27814.0 + inv * (-22569.18 + inv * 6838.66)))), 0, 255)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = 64  # gray for invalid
    return rgb.astype(np.uint8)


def colormap_conf(c: np.ndarray) -> np.ndarray:
    """Single-channel viridis-ish colormap on conf percentile range."""
    lo, hi = np.percentile(c, (2, 98))
    n = np.clip((c - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    r = np.clip(68 + 200 * n, 0, 255)
    g = np.clip(1 + 230 * n, 0, 255)
    b = np.clip(84 + (170 - 84) * n, 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def to_uint8_image(img_norm: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) in [-1, 1] → (H, W, 3) uint8."""
    arr = (img_norm[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    return (arr * 255).astype(np.uint8)


def hstack(*imgs: np.ndarray) -> np.ndarray:
    h = max(im.shape[0] for im in imgs)
    pads = []
    for im in imgs:
        if im.shape[0] < h:
            pad = np.zeros((h - im.shape[0], im.shape[1], 3), im.dtype)
            pads.append(np.concatenate([im, pad], axis=0))
        else:
            pads.append(im)
    return np.concatenate(pads, axis=1)


def vstack(*imgs: np.ndarray) -> np.ndarray:
    w = max(im.shape[1] for im in imgs)
    pads = []
    for im in imgs:
        if im.shape[1] < w:
            pad = np.zeros((im.shape[0], w - im.shape[1], 3), im.dtype)
            pads.append(np.concatenate([im, pad], axis=1))
        else:
            pads.append(im)
    return np.concatenate(pads, axis=0)


def label_strip(text: str, w: int) -> np.ndarray:
    """Black strip with white centred text — drawn cell-by-cell from a tiny font."""
    h = 18
    strip = np.zeros((h, w, 3), dtype=np.uint8)
    # Skip text rendering complexity — just return a uniform separator.
    strip[:] = (16, 16, 16)
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co3d-root", type=Path,
                    default=Path("/home/ttuser/experiments/vggt/co3d_data"))
    ap.add_argument("--category", default="apple")
    ap.add_argument("--seq", default="110_13051_23361")
    ap.add_argument("--frame-i", type=int, default=0)
    ap.add_argument("--frame-j", type=int, default=40)
    ap.add_argument("--img-h", type=int, default=512)
    ap.add_argument("--img-w", type=int, default=512)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/home/ttuser/experiments/mast3r/tt-metal/models/demos/mast3r/media"))
    args = ap.parse_args()

    if args.img_h % 32 or args.img_w % 32:
        ap.error(f"--img-h/--img-w must be divisible by 32 (got {args.img_h}x{args.img_w})")

    img_dir = args.co3d_root / args.category / args.seq / "images"
    frames = sorted(img_dir.glob("frame*.jpg"))
    if args.frame_i >= len(frames) or args.frame_j >= len(frames):
        ap.error(f"frame indices out of range; sequence has {len(frames)} frames")
    path_i = frames[args.frame_i]
    path_j = frames[args.frame_j]
    print(f"# images: {path_i.name}  +  {path_j.name}  → {args.img_h}x{args.img_w}")

    img_i, _ = load_image_for_dust3r(path_i, args.img_h, args.img_w)
    img_j, _ = load_image_for_dust3r(path_j, args.img_h, args.img_w)

    import ttnn
    from tt.ttnn_dust3r import dust3r_forward
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32 * 1024)
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()

    try:
        state = load_checkpoint()
        print("# loading reference DUSt3R")
        ref_model = load_dust3r(state).eval()

        print("# torch reference forward (CPU)")
        with torch.no_grad():
            ref1, ref2 = ref_model(img_i, img_j)

        print("# warming TT program cache")
        _ = dust3r_forward(img_i, img_j, state, device)

        print("# port forward on Blackhole")
        import time
        t0 = time.perf_counter()
        tt1, tt2 = dust3r_forward(img_i, img_j, state, device)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Per-head PCC + combined.
        p1 = pcc(ref1, tt1)
        p2 = pcc(ref2, tt2)
        p_combined = pcc(torch.stack([ref1, ref2]), torch.stack([tt1, tt2]))
        print(f"  port-vs-ref PCC head1={p1:.4f} head2={p2:.4f} combined={p_combined:.4f}")
        print(f"  latency: {latency_ms:.1f} ms")

        # Build visualisation panels.
        rgb_i = to_uint8_image(img_i)
        rgb_j = to_uint8_image(img_j)
        ref_z1 = activate_pts3d(ref1)[0, ..., 2].cpu().numpy()
        ref_z2 = activate_pts3d(ref2)[0, ..., 2].cpu().numpy()
        tt_z1 = activate_pts3d(tt1)[0, ..., 2].cpu().numpy()
        tt_z2 = activate_pts3d(tt2)[0, ..., 2].cpu().numpy()
        ref_c1 = activate_conf(ref1)[0].cpu().numpy()
        ref_c2 = activate_conf(ref2)[0].cpu().numpy()
        tt_c1 = activate_conf(tt1)[0].cpu().numpy()
        tt_c2 = activate_conf(tt2)[0].cpu().numpy()

        # Top row: input images + ref/port head1 depth + ref/port head1 conf
        # Bottom row: same but for head2 (which is the second view's pointmap in view-1's frame)
        sep = label_strip("", rgb_i.shape[1])
        row1 = hstack(rgb_i, colormap_depth(ref_z1), colormap_depth(tt_z1),
                      colormap_conf(ref_c1), colormap_conf(tt_c1))
        row2 = hstack(rgb_j, colormap_depth(ref_z2), colormap_depth(tt_z2),
                      colormap_conf(ref_c2), colormap_conf(tt_c2))
        canvas = vstack(row1, sep, row2)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / f"demo_{args.seq}_{args.frame_i}_{args.frame_j}_{args.img_h}x{args.img_w}.png"
        Image.fromarray(canvas).save(out_path)
        print(f"# wrote {out_path}")

        # Also write a short text caption next to it.
        cap = args.out_dir / f"{out_path.stem}.txt"
        cap.write_text(
            f"{args.seq} frames {args.frame_i}+{args.frame_j} @ {args.img_h}x{args.img_w}\n"
            f"port-vs-ref PCC: head1={p1:.4f}  head2={p2:.4f}  combined={p_combined:.4f}\n"
            f"latency: {latency_ms:.1f} ms (post-warmup)\n"
            f"layout: rows = view-1, view-2; cols = input | ref depth | port depth | ref conf | port conf\n"
        )
        print(f"# wrote {cap}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

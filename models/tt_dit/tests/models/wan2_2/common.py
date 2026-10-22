# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the WAN 2.2 pipeline tests."""

import numpy as np
import torch
from loguru import logger
from PIL import Image

import ttnn

# Upper timing limits in seconds per (mesh shape, height, arch): 30% headroom over measured, 100% on
# encoder. test_performance_wan.py gates on them (with TIMING_LOWER_FRACTION as the lower bound);
# test_pipeline_wan.py gates its cold single run on a multiple of the total.
T2V_TIMING_LIMITS = {
    ((2, 2), 480, "bh"): {"encoder": 0.12, "denoising": 680.0, "vae": 60.0, "total": 760.0},
    ((2, 4), 480, "bh"): {"encoder": 0.1, "denoising": 240.0, "vae": 5.0, "total": 255.0},
    ((2, 4), 480, "wh"): {"encoder": 0.1, "denoising": 800.0, "vae": 9.0, "total": 850.0},
    # 4x8: ring measurements only; linear needs remeasuring.
    ((4, 8), 480, "bh"): {"encoder": 0.21, "denoising": 75.0, "vae": 0.42, "total": 76.0},
    ((4, 8), 480, "wh"): {"encoder": 0.23, "denoising": 142.0, "vae": 1.55, "total": 144.0},
    ((4, 8), 720, "bh"): {"encoder": 0.22, "denoising": 205.0, "vae": 0.85, "total": 206.0},
    ((4, 8), 720, "wh"): {"encoder": 0.22, "denoising": 449.0, "vae": 3.4, "total": 452.0},
    ((4, 32), 720, "bh"): {"encoder": 0.54, "denoising": 75.0, "vae": 5.0, "total": 80.5},
}
# I2V configs measured on their own; anything missing falls back to the T2V limits (TODO: measure).
I2V_TIMING_LIMITS = {}
# BH 4x8 linear topology is expected to be slower than ring; relax the limits by this factor.
BH_4X8_LINEAR_SLACK = 1.10
# Uncalibrated placeholder (#60481): a run under this fraction of the upper limit is suspicious
# (skipped work, a stale cache hit) rather than fast.
TIMING_LOWER_FRACTION = 0.5


def timing_limits(model_type, mesh_shape, height, topology):
    """``{metric: (lower, upper)}`` seconds for this config, or None if it has no limits."""
    key = (tuple(mesh_shape), height, "bh" if ttnn.device.is_blackhole() else "wh")
    upper = (I2V_TIMING_LIMITS.get(key) if model_type == "i2v" else None) or T2V_TIMING_LIMITS.get(key)
    if upper is None:
        return None
    slack = BH_4X8_LINEAR_SLACK if tuple(mesh_shape) == (4, 8) and topology == ttnn.Topology.Linear else 1.0
    return {k: (v * slack * TIMING_LOWER_FRACTION, v * slack) for k, v in upper.items()}


def check_timings(measured, limits, label):
    """Messages for each measured metric outside its ``(lower, upper)`` limits."""
    return [
        f"{label} {k} = {measured[k]:.3f}s outside [{lower:.3f}, {upper:.3f}]s"
        for k, (lower, upper) in limits.items()
        if k in measured and not lower <= measured[k] <= upper
    ]


def create_fractal_image(width: int, height: int) -> Image.Image:
    """Deterministic synthetic conditioning image for the I2V tests (no image file needed in CI)."""
    c = np.linspace(-2.0, 1.0, width)[None, :] + 1j * np.linspace(-1.5, 1.5, height)[:, None]
    z = np.zeros_like(c)
    img = np.zeros(c.shape, dtype=np.uint8)
    for i in range(32):
        z = z * z + c
        img[(img == 0) & (np.abs(z) > 2)] = 255 - 8 * i
    return Image.fromarray(np.dstack((img, np.roll(img, width // 10, 1), np.roll(img, height // 10, 0))), "RGB")


def check_output_sanity(frames, *, num_frames, height, width, log=True):
    """Guard against a distributed pipeline that 'runs' but emits corrupt/blank/frozen frames.

    Cheap, reference-free statistical checks (shape, finiteness, range, spatial variance, temporal
    motion) -- not a full quality gate. Thresholds sit well below any real video so they only fire
    on genuinely broken output, never on run-to-run noise.

    Args:
        frames: video array/tensor of shape (num_frames, height, width, 3), uint8, batch dim removed.
        num_frames, height, width: expected output geometry.
    """
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    frames = np.asarray(frames)

    # Geometry: the distributed VAE decode must produce the full requested video.
    expected_shape = (num_frames, height, width, 3)
    assert frames.shape == expected_shape, f"Unexpected output shape {frames.shape}, expected {expected_shape}"

    # No NaN/Inf. uint8 can't carry them, but guard in case the pipeline hands back a float buffer.
    if np.issubdtype(frames.dtype, np.floating):
        assert np.isfinite(frames).all(), "Output video contains NaN/Inf"

    vmin, vmax = int(frames.min()), int(frames.max())
    assert 0 <= vmin and vmax <= 255, f"Output outside uint8 range: [{vmin}, {vmax}]"

    # Not a flat/blank buffer (all-black or a single constant colour) -- a common corruption mode.
    global_std = float(frames.std())
    assert global_std > 1.0, f"Output has near-zero variance (std={global_std:.3f}); frames look blank/corrupt"

    # Temporal motion: the video must animate, not repeat one static frame.
    # int16 cast avoids uint8 wraparound in the difference.
    mean_frame_delta = float(np.abs(np.diff(frames.astype(np.int16), axis=0)).mean())
    assert (
        mean_frame_delta > 0.5
    ), f"Consecutive frames are near-identical (mean delta={mean_frame_delta:.3f}); video appears frozen/static"

    if log:
        logger.info(
            f"Output sanity OK: shape={frames.shape}, range=[{vmin},{vmax}], "
            f"std={global_std:.2f}, mean_frame_delta={mean_frame_delta:.2f}"
        )


def check_first_frame_matches_seed(frames, *, seed_image, width, height, pcc_floor=0.3):
    """Seed-agnostic correctness signal for I2V: decoded frame 0 must resemble the seed image.

    The pipeline conditions frame 0 on the seed image (VAE-encodes it into the first latent frame),
    so the decoded first frame must strongly resemble the seed. This catches a broken
    image-conditioning path, and unlike VBench it works regardless of seed content (fractal or
    natural). VAE round-trip + denoising means it won't be pixel-identical, so we gate on
    correlation, not equality.

    Args:
        frames: decoded video array/tensor, shape (num_frames, height, width, 3), batch dim removed.
        seed_image: the PIL seed image conditioned into frame 0.
        width, height: target resolution; the seed is resized to this before comparison.
        pcc_floor: minimum Pearson correlation. Provisional floor -- catches a totally-broken
            conditioning path (near-zero correlation); a healthy round-trip correlates well above it.
            Tighten once real values are observed.
    """
    f0 = frames[0]
    if isinstance(f0, torch.Tensor):
        f0 = f0.cpu().numpy()
    f0 = np.asarray(f0).astype(np.float64)
    seed = np.asarray(seed_image.convert("RGB").resize((width, height))).astype(np.float64)
    assert f0.shape == seed.shape, f"frame-0 shape {f0.shape} != seed shape {seed.shape}"
    pcc = float(np.corrcoef(f0.ravel(), seed.ravel())[0, 1])
    logger.info(f"I2V frame-0 vs seed-image correlation (PCC) = {pcc:.4f}")
    assert pcc > pcc_floor, (
        f"Decoded frame 0 barely correlates with the seed image (PCC={pcc:.3f}); "
        "the I2V image-conditioning path is likely broken"
    )

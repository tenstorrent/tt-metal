# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the WAN 2.2 pipeline tests."""

import numpy as np
import torch
from loguru import logger
from PIL import Image

import ttnn

# (lower, upper) timing limits in seconds per (model, mesh shape, height, arch), from the CI perf
# logs of the 2026-10-18..22 nightlies plus one branch dispatch: lower is 0.7x the fastest and upper
# 1.3x the slowest observation across hosts, sc4 quads and ranks (encoder, being tiny and noisy:
# 0.5x / 2x). The spread is real: bh_sc1 hosts differ by ~40% on denoising, and the two bh_sc4
# quads by ~25%. test_performance_wan.py gates on both bounds; test_pipeline_wan.py gates its cold
# single run on a multiple of the total upper limit.
#
# PENDING: the BH galaxy (4x8 BH, 4x32) bands were measured on the standard bh_sc1 / bh_sc4 pools;
# their CI legs now run on the high-power pools and get re-derived from those runs. The BH 4x8 vae
# bounds also predate the bh_sc1 host-contention fix (#61784, ~0.4 s off the readback): they keep the
# pre-regression upper limit and an estimated lower until then.
TIMING_LIMITS = {
    ("t2v", (2, 2), 480, "bh"): {
        "encoder": (0.01, 0.16),
        "denoising": (270.0, 530.0),
        "vae": (2.0, 4.0),
        "total": (270.0, 530.0),
    },
    ("t2v", (4, 8), 480, "bh"): {
        "encoder": (0.035, 0.17),
        "denoising": (27.0, 74.0),
        "vae": (0.23, 0.42),  # PENDING
        "total": (27.0, 75.0),
    },
    ("t2v", (4, 8), 480, "wh"): {
        "encoder": (0.037, 0.17),
        "denoising": (75.0, 150.0),
        "vae": (0.77, 1.7),
        "total": (76.0, 150.0),
    },
    ("t2v", (4, 8), 720, "bh"): {
        "encoder": (0.034, 0.17),
        "denoising": (86.0, 210.0),
        "vae": (0.42, 0.85),  # PENDING
        "total": (87.0, 210.0),
    },
    ("t2v", (4, 8), 720, "wh"): {
        "encoder": (0.038, 0.17),
        "denoising": (240.0, 450.0),
        "vae": (1.6, 3.4),
        "total": (240.0, 460.0),
    },
    # 4x32 T2V has only run on the slower sc4 quad; its lower bounds take the faster quad's I2V
    # times (same DiT and VAE decode).
    ("t2v", (4, 32), 720, "bh"): {
        "encoder": (0.058, 0.29),
        "denoising": (24.0, 59.0),
        "vae": (0.28, 1.2),
        "total": (25.0, 60.0),
    },
    ("i2v", (4, 32), 720, "bh"): {
        "encoder": (0.058, 0.29),
        "denoising": (24.0, 59.0),
        "vae": (0.28, 1.2),
        "total": (25.0, 62.0),
    },
    # One observation (a slow bh_sc1 host); the band follows T2V's so a fast host passes too.
    ("i2v", (4, 8), 720, "bh"): {
        "encoder": (0.041, 0.17),
        "denoising": (86.0, 210.0),
        "vae": (0.42, 0.85),  # PENDING
        "total": (87.0, 210.0),
    },
    # Not run in CI, so never measured here: upper limits only, carried over from before.
    ("t2v", (2, 4), 480, "bh"): {
        "encoder": (0.0, 0.1),
        "denoising": (0.0, 240.0),
        "vae": (0.0, 5.0),
        "total": (0.0, 255.0),
    },
    ("t2v", (2, 4), 480, "wh"): {
        "encoder": (0.0, 0.1),
        "denoising": (0.0, 800.0),
        "vae": (0.0, 9.0),
        "total": (0.0, 850.0),
    },
}
# BH 4x8 linear topology is expected to be slower than ring; relax the upper limits by this factor.
BH_4X8_LINEAR_SLACK = 1.10


def timing_limits(model_type, mesh_shape, height, topology):
    """``{metric: (lower, upper)}`` seconds for this config, or None if it has no limits.

    I2V configs without their own entry use the T2V limits.
    """
    key = (tuple(mesh_shape), height, "bh" if ttnn.device.is_blackhole() else "wh")
    limits = TIMING_LIMITS.get((model_type, *key)) or TIMING_LIMITS.get(("t2v", *key))
    if limits is None:
        return None
    slack = BH_4X8_LINEAR_SLACK if tuple(mesh_shape) == (4, 8) and topology == ttnn.Topology.Linear else 1.0
    return {k: (lower, upper * slack) for k, (lower, upper) in limits.items()}


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

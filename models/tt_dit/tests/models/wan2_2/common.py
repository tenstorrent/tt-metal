# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the WAN 2.2 pipeline tests."""

import math
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image


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


# The official I2V example (Wan-Video/Wan2.2 README and wan/configs/wan_i2v_A14B.py). Image, prompt,
# size, flow shift, boundary and seed match the diffusers reference run, and so does the noise: both
# draw it from the CPU generator.
OFFICIAL_IMAGE_URL = (
    "https://raw.githubusercontent.com/Wan-Video/Wan2.2/1ea34ff48f87168174e12956e200b1d908b1c5ff/examples/i2v_input.JPG"
)
OFFICIAL_IMAGE_SHA256 = "077e3d965090c9028c69c00931675f42e1acc815c6eb450ab291b3b72d211a8e"
OFFICIAL_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline "
    "gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring "
    "crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally "
    "relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's "
    "intricate details and the refreshing atmosphere of the seaside."
)
OFFICIAL_FLOW_SHIFT = 5.0
OFFICIAL_BOUNDARY_RATIO = 0.9  # the checkpoint's model_index.json
OFFICIAL_SEED = 0


def official_image() -> Image.Image:
    path = Path(torch.hub.get_dir()) / "wan2.2_i2v_input.jpg"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(OFFICIAL_IMAGE_URL, str(path), hash_prefix=OFFICIAL_IMAGE_SHA256)
    return Image.open(path).convert("RGB")


def official_size(max_area: int, image: Image.Image) -> tuple[int, int]:
    """Wan's image2video.py: the image's aspect ratio at max_area, in whole latent patches."""
    aspect = image.height / image.width
    lat_h = round(math.sqrt(max_area * aspect) // 8 // 2 * 2)
    lat_w = round(math.sqrt(max_area / aspect) // 8 // 2 * 2)
    return lat_w * 8, lat_h * 8


def official_i2v_setup(max_area: int) -> tuple[Image.Image, int, int]:
    """The official image resized to its aspect ratio at max_area, with that size."""
    image = official_image()
    width, height = official_size(max_area, image)
    return image.resize((width, height)), width, height

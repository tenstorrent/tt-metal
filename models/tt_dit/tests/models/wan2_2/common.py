# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the WAN 2.2 pipeline tests."""

import numpy as np
import torch
from loguru import logger


def check_output_sanity(frames, *, num_frames, height, width):
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

    logger.info(
        f"Output sanity OK: shape={frames.shape}, range=[{vmin},{vmax}], "
        f"std={global_std:.2f}, mean_frame_delta={mean_frame_delta:.2f}"
    )

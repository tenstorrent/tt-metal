# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Checks of one image's input to the MiniMax-M3 vision encoder.

The input is what the image processor gives for an image of h x w patches of 14 x 14 pixels. pixel_values holds the
(h * w, 1176) float patch rows, and image_grid_thw is the (1, 3) grid [[1, h, w]]. h and w are even, as the projector
merges each 2 x 2 block of patches into one vision token. This module needs only torch, not ttnn.
"""
from __future__ import annotations

from typing import Sequence, Union

import torch

MERGE_SIZE = 2  # each MERGE_SIZE x MERGE_SIZE block of patches becomes one vision token


def _holds_integers(grid: torch.Tensor) -> bool:
    """True for a tensor of an integer dtype: not bool, float or complex."""
    return not (grid.is_floating_point() or grid.is_complex() or grid.dtype == torch.bool)


def num_tokens(grid_thw: Union[torch.Tensor, Sequence[int]]) -> int:
    """Number of vision tokens of one image: h * w / 4 for its patch grid (1, h, w).

    grid_thw: a tensor or sequence of shape (1, 3) or (3,). Raises TypeError unless it holds integers. Raises
    ValueError unless it is one grid with t = 1 (no video) and h and w positive multiples of 2.
    """
    grid = torch.as_tensor(grid_thw)
    if not _holds_integers(grid):
        raise TypeError(f"grid_thw must hold integers, got {grid.dtype}")
    if grid.numel() != 3:
        raise ValueError(f"grid_thw must be the (t, h, w) grid of one image, got shape {tuple(grid.shape)}")
    t, h, w = (int(x) for x in grid.flatten().tolist())
    if t != 1:
        raise ValueError(f"grid_thw must have t = 1 (one image, no video), got (t, h, w) = {(t, h, w)}")
    if h < MERGE_SIZE or w < MERGE_SIZE or h % MERGE_SIZE or w % MERGE_SIZE:
        raise ValueError(f"grid_thw h and w must be positive multiples of {MERGE_SIZE}, got (t, h, w) = {(t, h, w)}")
    return h * w // MERGE_SIZE**2


def check_image_grid(grid_thw: torch.Tensor) -> int:
    """Check one image's grid and return its patch count, h * w.

    Raises ValueError unless grid_thw is an integer tensor [[1, h, w]] of shape (1, 3), with h and w positive
    multiples of 2.
    """
    if not isinstance(grid_thw, torch.Tensor):
        raise ValueError(f"grid_thw must be a torch tensor, got {type(grid_thw).__name__}")
    if not _holds_integers(grid_thw) or grid_thw.shape != (1, 3):
        raise ValueError(
            "grid_thw must be the grid of one image, an integer tensor (1, 3),"
            f" got {grid_thw.dtype} {tuple(grid_thw.shape)}"
        )
    return num_tokens(grid_thw) * MERGE_SIZE**2  # num_tokens checks t, h and w


def check_image_input(pixel_values: torch.Tensor, grid_thw: torch.Tensor, patch_dim: int) -> None:
    """Check one image's input in the form the image processor gives it.

    Raises ValueError unless grid_thw passes check_image_grid and pixel_values are float patch rows (h * w, patch_dim).
    Their fp32 sum must also be finite, so no value is NaN or Inf. patch_dim is the config's, 1176 for MiniMax-M3.
    The check does not limit the image size.
    """
    if not isinstance(pixel_values, torch.Tensor) or not isinstance(grid_thw, torch.Tensor):
        raise ValueError(
            "pixel_values and grid_thw must be torch tensors,"
            f" got {type(pixel_values).__name__} and {type(grid_thw).__name__}"
        )
    patches = check_image_grid(grid_thw)
    if not pixel_values.is_floating_point() or pixel_values.ndim != 2 or pixel_values.shape[1] != patch_dim:
        raise ValueError(
            f"pixel_values must be float patch rows (num_patches, {patch_dim}),"
            f" got {pixel_values.dtype} {tuple(pixel_values.shape)}"
        )
    if pixel_values.shape[0] != patches:
        raise ValueError(
            f"{pixel_values.shape[0]} patch rows for grid_thw {grid_thw.tolist()}, which has {patches} patches"
        )
    # A NaN or Inf value makes the sum NaN or Inf, and one sum costs much less than isfinite().all(). Finite values
    # fail only if their sum overflows fp32, far beyond the processor's range of about [-1.8, 2.2].
    total = pixel_values.sum(dtype=torch.float32)
    if not torch.isfinite(total):
        raise ValueError(f"pixel_values must be finite, got a sum of {total.item()} in fp32")

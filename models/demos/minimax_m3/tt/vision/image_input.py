# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
The input of one image to the MiniMax-M3 vision encoder, and its check.

The checkpoint's image processor (MiniMaxM3VLImageProcessor, transformers 5.12.1) cuts an image into patches of
14 x 14 pixels. For an image of (1, h, w) patches it gives pixel_values, (h * w, 1176) float patch rows of 3 channels
x 2 frames x 14 x 14 values (the vision config's patch_dim), and image_grid_thw, the (1, 3) grid [[1, h, w]] with h and
w multiples of 2. The projector merges each 2 x 2 block of patches into one vision token, so the image has h * w / 4
tokens.

It needs only torch, so code that prepares images on a host without ttnn can use the same rules.
"""
from __future__ import annotations

from typing import Sequence, Union

import torch

MERGE_SIZE = 2  # each MERGE_SIZE x MERGE_SIZE block of patches becomes one vision token


def _holds_integers(grid: torch.Tensor) -> bool:
    """True for a tensor of an integer dtype: not bool, float or complex."""
    return not (grid.is_floating_point() or grid.is_complex() or grid.dtype == torch.bool)


def num_tokens(grid_thw: Union[torch.Tensor, Sequence[int]]) -> int:
    """Vision tokens of one image, h * w / 4 for its patch grid (1, h, w): the rows of the model's output for it.

    grid_thw: the grid of one image, (1, 3) or (3,). Raises TypeError unless it holds integers, and ValueError unless
    t is 1 (video is not supported) and h and w are positive multiples of 2.
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
    """Raise ValueError unless grid_thw is the grid of one image, an integer tensor (1, 3) [[1, h, w]] with h and w
    positive multiples of 2. Returns its patch count, h * w.

    check_image_input checks the grid with it, and so does M3VLVisionModel.forward_device, which then compares the patch
    count with the rows of the pixel values already on the device.
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
    """Raise ValueError unless pixel_values and grid_thw are the input of one image in the form the image processor
    gives it: pixel_values float patch rows (h * w, patch_dim) with no NaN or Inf value (their sum in fp32 must be
    finite), and grid_thw a grid that check_image_grid accepts. patch_dim is the number of values per patch in the
    model's vision config (MiniMaxM3VisionConfig.patch_dim, 1176). It does not bound the image's size: any such grid
    passes, also one outside the 16 to 2304 patches that the processor gives by default.

    M3VLVisionModel.push_pixel_values runs it before any device work, as the device checks neither that the rows match
    the grid nor that the values are finite.
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
    # One NaN or Inf value makes the sum NaN or Inf, and one fp32 sum costs far less than isfinite().all(). fp16 rows
    # cannot overflow it. Finite values fail only when their sum overflows fp32, which at 2304 x 1176 values takes about
    # 1e32 each (3.4e38 / 2.7e6); the processor's values lie within [-1.8, 2.2].
    total = pixel_values.sum(dtype=torch.float32)
    if not torch.isfinite(total):
        raise ValueError(f"pixel_values must be finite, got a sum of {total.item()} in fp32")

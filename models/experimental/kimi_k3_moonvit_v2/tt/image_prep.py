# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Image pre-processing on the host: a PIL image to `(pixel_values, grid_thws)`, as Kimi-K3's HF
processor (`reference/kimi_k3_vision_processing.py`) produces them.

This is layer 1 of the padding: the image is padded with black on the bottom and right to a multiple
of `merge_kernel_size * patch_size` (28 px), and after normalization that black is -1.0. Those pad
patches are real tokens. Layer 2, the zero rows up to `max_patches`, is `host_prep.prepare_host_inputs`.

Only RGB images are accepted; every other mode is rejected rather than converted, so transparency
(which HF composites onto a chessboard) is never silently dropped. Images over `config.max_patches`
are rejected before the resize, never shrunk to fit. `tests/test_image_prep.py` checks the rest bit
for bit against the HF copy.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image

from .model_config import MoonViTConfig, check_grid

# `image_mean` and `image_std` of `preprocessor_config.json`, the same for every channel.
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5


class ResizePlan(NamedTuple):
    new_w: int
    new_h: int
    pad_w: int
    pad_h: int
    grid_h: int
    grid_w: int


def resize_plan(width: int, height: int, config: MoonViTConfig) -> ResizePlan:
    """HF's `navit_resize_image`: shrink (never enlarge) to the patch limits, then pad each side to a
    multiple of one merge window."""
    p, side = config.patch_size, config.max_side_patches
    s1 = math.sqrt(config.in_patch_limit / (max(1.0, width // p) * max(1.0, height // p)))
    scale = min(1.0, s1, side * p / width, side * p / height)
    new_w = min(max(1, int(width * scale)), side * p)
    new_h = min(max(1, int(height * scale)), side * p)
    fh, fw = config.merge_kernel_size[0] * p, config.merge_kernel_size[1] * p
    pad_h, pad_w = (fh - new_h % fh) % fh, (fw - new_w % fw) % fw
    return ResizePlan(new_w, new_h, pad_w, pad_h, (new_h + pad_h) // p, (new_w + pad_w) // p)


def image_to_patches(image: Image.Image, config: MoonViTConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """An RGB PIL image -> `pixel_values` (h*w, 3, p, p) float32 in raster order and `grid_thws` [[1, h, w]] int64.

    Other modes are rejected: convert them to RGB first, knowing that HF composites alpha onto a chessboard
    and converts after the resize, so a converted image matches HF only if it had no alpha and isn't resized.
    """
    if image.mode != "RGB":
        raise ValueError(f"image mode {image.mode!r} is not supported, only 'RGB'")
    plan = resize_plan(*image.size, config)
    grid_thws = torch.tensor([[1, plan.grid_h, plan.grid_w]], dtype=torch.int64)
    try:
        check_grid(grid_thws, config)
    except ValueError as e:
        raise ValueError(f"image {image.width}x{image.height}: {e}") from e

    image = image.resize((plan.new_w, plan.new_h), resample=Image.Resampling.BICUBIC)
    x = np.pad(
        np.asarray(image),
        ((0, plan.pad_h), (0, plan.pad_w), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    x = (x / 255.0).astype(np.float32)
    x -= IMAGE_MEAN
    x *= 1.0 / IMAGE_STD
    p = config.patch_size
    x = x.reshape(plan.grid_h, p, plan.grid_w, p, 3).transpose(0, 2, 4, 1, 3).reshape(-1, 3, p, p)
    return torch.from_numpy(x), grid_thws

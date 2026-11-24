# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Image pre-processing on the host: a PIL image to `(pixel_values, grid_thws)`, as Kimi-K3's HF
processor (`reference/kimi_k3_vision_processing.py`) produces them. Two steps: `resize_image` (resize and
pad, uint8) and `pixels_to_patches` (normalize and patchify, float32); `image_to_patches` runs both.
`resize_array` is `resize_image` for a uint8 array; an image that needs no resize skips PIL.

This is layer 1 of the padding: the image is padded with black on the bottom and right to a multiple
of `merge_kernel_size * patch_size` (28 px), and after normalization that black is -1.0. Those pad
patches are real tokens. Layer 2, the zero rows up to `max_patches`, is `host_prep.prepare_host_inputs`.

Only RGB images are accepted; every other mode is rejected rather than converted, so transparency
(which HF composites onto a chessboard) is never silently dropped. The default `config.max_patches`
holds the grid of every image size; with a smaller one, an image over it is rejected before the
resize, never shrunk to fit. `tests/test_image_prep.py` checks the rest bit for bit against the HF copy.
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


def resize_image(image: Image.Image, config: MoonViTConfig) -> np.ndarray:
    """An RGB PIL image -> uint8 [H, W, 3], resized and padded with black as HF's processor does. H and W are
    multiples of one merge window (28 px), so the image is whole patches and whole merge windows.

    Other modes are rejected: convert them to RGB first, knowing that HF composites alpha onto a chessboard
    and converts after the resize, so a converted image matches HF only if it had no alpha and isn't resized.
    """
    if image.mode != "RGB":
        raise ValueError(f"image mode {image.mode!r} is not supported, only 'RGB'")
    return _resize_and_pad(image, image.width, image.height, config)


def resize_array(media: np.ndarray, config: MoonViTConfig) -> np.ndarray:
    """`resize_image` for uint8 RGB pixels [height, width, 3]. An image that needs no resize never goes through PIL
    (at scale 1 Pillow's resize returns a copy, so this is the same), and is copied once, into the padded array."""
    if media.dtype != np.uint8 or media.ndim != 3 or media.shape[2] != 3 or 0 in media.shape:
        raise ValueError(f"expected a non-empty uint8 RGB [height, width, 3] array, got {media.dtype} {media.shape}")
    return _resize_and_pad(media, media.shape[1], media.shape[0], config)


def _resize_and_pad(image: Image.Image | np.ndarray, width: int, height: int, config: MoonViTConfig) -> np.ndarray:
    """The steps `resize_image` and `resize_array` share, for an RGB image of width x height: check the grid, resize
    with Pillow only if the size changes, and pad."""
    plan = resize_plan(width, height, config)
    try:
        check_grid(torch.tensor([[1, plan.grid_h, plan.grid_w]]), config)
    except ValueError as e:
        raise ValueError(f"image {width}x{height}: {e}") from e
    if (plan.new_w, plan.new_h) != (width, height):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.resize((plan.new_w, plan.new_h), resample=Image.Resampling.BICUBIC)
    return _pad(np.asarray(image), plan)


def _pad(pixels: np.ndarray, plan: ResizePlan) -> np.ndarray:
    """`np.pad` with black on the bottom and right, zeroing only the pad."""
    height, width = pixels.shape[:2]
    out = np.empty((height + plan.pad_h, width + plan.pad_w, 3), np.uint8)
    out[:height, :width] = pixels
    out[:height, width:] = 0
    out[height:] = 0
    return out


def check_pixels(pixels: np.ndarray, config: MoonViTConfig) -> tuple[int, int]:
    """The patch grid (h, w) of `resize_image`'s output; ValueError unless it is uint8 [H, W, 3] with H and W multiples
    of one merge window, and the grid is within the config's limits (`check_grid`)."""
    fh, fw = (k * config.patch_size for k in config.merge_kernel_size)
    shape = pixels.shape
    if pixels.dtype != np.uint8 or len(shape) != 3 or shape[2] != 3 or shape[0] % fh or shape[1] % fw:
        raise ValueError(f"expected uint8 [H, W, 3], H a multiple of {fh} and W of {fw}; got {pixels.dtype} {shape}")
    h, w = shape[0] // config.patch_size, shape[1] // config.patch_size
    check_grid(torch.tensor([[1, h, w]]), config)
    return h, w


def pixels_to_patches(pixels: np.ndarray, config: MoonViTConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """`resize_image`'s uint8 [H, W, 3] -> `pixel_values` (h*w, 3, p, p) float32 in raster order and `grid_thws`
    [[1, h, w]] int64: normalized and cut into patches as HF's processor does."""
    h, w = check_pixels(pixels, config)
    grid_thws = torch.tensor([[1, h, w]], dtype=torch.int64)
    p = config.patch_size

    x = (pixels / 255.0).astype(np.float32)
    x -= IMAGE_MEAN
    x *= 1.0 / IMAGE_STD
    x = x.reshape(h, p, w, p, 3).transpose(0, 2, 4, 1, 3).reshape(-1, 3, p, p)
    return torch.from_numpy(x), grid_thws


def image_to_patches(image: Image.Image, config: MoonViTConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """An RGB PIL image -> `pixel_values` (h*w, 3, p, p) float32 in raster order and `grid_thws` [[1, h, w]] int64:
    `resize_image`, then `pixels_to_patches`."""
    return pixels_to_patches(resize_image(image, config), config)

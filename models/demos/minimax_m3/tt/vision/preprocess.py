# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host side of the MiniMax-M3 vision encoder's input: checks a resized image and turns it into patch rows.

The image is what the media module's prepare returns (M3Image.pixels). patch_rows gives the pixel_values and
image_grid_thw of Hugging Face's MiniMaxM3VLImageProcessor, bit for bit with the versions tt-metal pins. It copies the
processor's steps so that transformers is not imported. Needs torch, torchvision and numpy, not ttnn.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

from models.demos.minimax_m3.tt.vision.image_input import MERGE_SIZE

# The image processor's settings, which are also the vision config's patch geometry. The encoder checks its config
# against them, and the media module's resize uses them, so the images it gives pass check_pixels.
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
NUM_CHANNELS = 3
PATCH_DIM = NUM_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE  # 1176 values per patch row
FACTOR = PATCH_SIZE * MERGE_SIZE  # 28: the pixels of one vision token along each side
MIN_PIXELS = 56 * 56
MAX_PIXELS = 672 * 672
MAX_TOKENS = MAX_PIXELS // FACTOR**2  # 576
MAX_RATIO = 200  # the most stretched image the resize takes: its long side over its short one
# The most stretched image the resize gives, as long side over short side in tokens. Rounding can stretch an image
# past its own ratio: 9600 x 48 pixels (200:1, the most the resize takes) become 9492 x 28 (339:1).
MAX_RESIZED_ASPECT_RATIO = 339
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

# The processor folds the 1/255 rescale into the mean and std, in fp32; the same tensors give the same results.
_RESCALE_FACTOR = 1 / 255
_MEAN_255 = torch.tensor(IMAGE_MEAN) * (1.0 / _RESCALE_FACTOR)
_STD_255 = torch.tensor(IMAGE_STD) * (1.0 / _RESCALE_FACTOR)


def check_size(height: int, width: int) -> Tuple[int, int]:
    """Token grid (grid_h, grid_w) of a resized image of height x width pixels.

    Raises ValueError unless height and width are positive multiples of 28, the image has 4 to 576 tokens, and its
    aspect ratio is at most MAX_RESIZED_ASPECT_RATIO.
    """
    if height < FACTOR or width < FACTOR or height % FACTOR or width % FACTOR:
        raise ValueError(
            f"pixels height and width must be positive multiples of {FACTOR}, got height {height} and width {width}"
        )
    grid_h, grid_w = height // FACTOR, width // FACTOR
    if not MIN_PIXELS <= height * width <= MAX_PIXELS:
        raise ValueError(
            f"pixels must have {MIN_PIXELS // FACTOR**2} to {MAX_TOKENS} vision tokens of {FACTOR} x {FACTOR} pixels,"
            f" got {grid_h * grid_w} ({grid_h} down, {grid_w} across) for height {height} and width {width}"
        )
    if max(grid_h, grid_w) > MAX_RESIZED_ASPECT_RATIO * min(grid_h, grid_w):
        raise ValueError(
            f"pixels aspect ratio must be at most {MAX_RESIZED_ASPECT_RATIO}, the most the resize gives,"
            f" got height {height} and width {width}"
        )
    return grid_h, grid_w


def check_pixels(pixels: np.ndarray) -> Tuple[int, int]:
    """Check that pixels is a resized image and return its token grid (grid_h, grid_w).

    Raises TypeError unless pixels is a uint8 numpy array, and ValueError unless it is (height, width, 3) of a size
    that check_size takes.
    """
    if not isinstance(pixels, np.ndarray):
        raise TypeError(f"pixels must be a numpy array, got {type(pixels).__name__}")
    if pixels.dtype != np.uint8:
        raise TypeError(f"pixels must be uint8, got {pixels.dtype}")
    if pixels.ndim != 3 or pixels.shape[2] != NUM_CHANNELS:
        raise ValueError(f"pixels must be an RGB image (height, width, 3), got shape {pixels.shape}")
    return check_size(*pixels.shape[:2])


def patch_rows(pixels: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn a resized image into the encoder's patch rows, as the image processor does.

    It normalizes the pixels with the CLIP mean and std in fp32, and repeats the image as both frames of a temporal
    patch. It cuts the image into 14 x 14 patches, with each 2 x 2 block of patches next to each other.

    pixels: as prepare returns it (M3Image.pixels). A read-only array is copied first.
    Returns pixel_values (h * w, 1176) float32 and grid_thw [[1, h, w]] int64, with h = height / 14 and w = width / 14.
    Raises check_pixels' errors before any other work.
    """
    grid_h, grid_w = check_pixels(pixels)
    image = torch.from_numpy(np.require(pixels, requirements=("C_CONTIGUOUS", "WRITEABLE")))
    image = tvF.normalize(image.permute(2, 0, 1).to(torch.float32), _MEAN_255, _STD_255)  # (3, height, width)
    m, p = MERGE_SIZE, PATCH_SIZE
    patches = image.reshape(NUM_CHANNELS, grid_h, m, p, grid_w, m, p)
    patches = patches.permute(1, 4, 2, 5, 0, 3, 6)  # (grid_h, grid_w, m, m, channel, p, p)
    # The image as both frames of the temporal patch, (grid_h, grid_w, m, m, channel, frame, p, p): a view, so the
    # reshape below is the only copy.
    patches = patches.unsqueeze(5).expand(-1, -1, -1, -1, -1, TEMPORAL_PATCH_SIZE, -1, -1)
    h, w = grid_h * m, grid_w * m
    return patches.reshape(h * w, PATCH_DIM), torch.tensor([[1, h, w]], dtype=torch.int64)

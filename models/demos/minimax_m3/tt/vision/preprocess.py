# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
The host side of the MiniMax-M3 vision encoder's input: a resized image, as the media module's prepare gives it
(vision/media.py, M3Image.pixels), checked and turned into the encoder's patch rows. It needs torch, torchvision and
numpy, not ttnn.

  check_pixels(pixels) -> (grid_h, grid_w)
    The image must be one that prepare can give: uint8 (height, width, 3) with sides that are multiples of 28 pixels
    (2 x 2 patches of 14), 4 to 576 vision tokens and an aspect ratio of at most MAX_RESIZED_ASPECT_RATIO. grid_h and
    grid_w count its blocks of 28 x 28 pixels, the vision tokens down and across it.
  patch_rows(pixels) -> (pixel_values, grid_thw)
    1. Rescale to [0, 1] and normalize with the CLIP mean and std, in fp32, with the processor's fused arithmetic.
    2. The image as both frames of a temporal patch, cut into 14 x 14 patches with each 2 x 2 block of patches
       consecutive (the projector merges each block into one vision token): one row of 3 channels x 2 frames x 14 x 14
       values per patch.
    pixel_values (h * w, 1176) float32 and grid_thw [[1, h, w]] int64, in patches (h = height / 14, w = width / 14):
    what the checkpoint's Hugging Face image processor (MiniMaxM3VLImageProcessor, transformers 5.12.1) returns for
    the image.

This copies the processor's steps rather than calling it, which would import transformers, seconds more of start-up.
tests/unit/test_vision_preprocess_vs_hf.py checks prepare and patch_rows against the processor, bit for bit, with
torch 2.11.0, torchvision 0.26.0 and Pillow 12.3.0, the versions tt-metal pins.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

from models.demos.minimax_m3.tt.vision.image_input import MERGE_SIZE

# The image processor's settings, which are also the vision config's patch geometry. TtMiniMaxM3VisionEncoder checks
# them against the model's config when it is built, and the media module's resize takes them from here, so the
# images it gives are the ones check_pixels takes.
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
NUM_CHANNELS = 3
PATCH_DIM = NUM_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE  # 1176 values per patch row
FACTOR = PATCH_SIZE * MERGE_SIZE  # 28: the pixels of one vision token along each side
MIN_PIXELS = 56 * 56
MAX_PIXELS = 672 * 672
MAX_TOKENS = MAX_PIXELS // FACTOR**2  # 576
MAX_RATIO = 200  # the most stretched image the resize takes: its long side over its short one
# The most stretched resized image, in tokens along its long side per token along its short one. The rounding of the
# sides can stretch an image beyond its own aspect ratio: 9600 x 48 pixels (width x height; 200:1, the most the
# resize takes) become 9492 x 28 (339:1). With a short side of one token the long side is at most
# floor(sqrt(MAX_PIXELS * 200) / 28) = 339 tokens, and with two it is at most 288.
MAX_RESIZED_ASPECT_RATIO = 339
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

# The processor folds the 1/255 rescale into the mean and std, in fp32; the same tensors give the same results.
_RESCALE_FACTOR = 1 / 255
_MEAN_255 = torch.tensor(IMAGE_MEAN) * (1.0 / _RESCALE_FACTOR)
_STD_255 = torch.tensor(IMAGE_STD) * (1.0 / _RESCALE_FACTOR)


def check_pixels(pixels: np.ndarray) -> Tuple[int, int]:
    """The token grid (grid_h, grid_w) of a resized image, after checking that it is one prepare can return.

    Raises TypeError unless pixels is a uint8 numpy array, and ValueError unless it is (height, width, 3) with height
    and width positive multiples of 28, an area of 4 to 576 tokens (56 x 56 to 672 x 672 pixels) and an aspect ratio
    of at most MAX_RESIZED_ASPECT_RATIO: every image prepare returns passes, and nothing else the vision encoder has
    not been built for does.
    """
    if not isinstance(pixels, np.ndarray):
        raise TypeError(f"pixels must be a numpy array, got {type(pixels).__name__}")
    if pixels.dtype != np.uint8:
        raise TypeError(f"pixels must be uint8, got {pixels.dtype}")
    if pixels.ndim != 3 or pixels.shape[2] != NUM_CHANNELS:
        raise ValueError(f"pixels must be an RGB image (height, width, 3), got shape {pixels.shape}")
    height, width = pixels.shape[:2]
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


def patch_rows(pixels: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """The patch rows of a resized image (steps 1 and 2 of the module docstring): (pixel_values, grid_thw), what the
    processor returns for the image. pixel_values is (h * w, 1176) float32 and grid_thw the (1, 3) int64 grid
    [[1, h, w]], in patches of 14 x 14 pixels: twice grid_h and grid_w.

    pixels: as prepare returns it (M3Image.pixels); raises check_pixels' errors before any other work. A read-only
    array is copied first, which torch needs to take it without a warning.
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

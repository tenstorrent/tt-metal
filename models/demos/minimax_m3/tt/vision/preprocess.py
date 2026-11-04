# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
The CPU side of the MiniMax-M3 vision encoder: an image resized to the size the encoder takes, and the resized
pixels turned into the encoder's patch rows. It needs torch, torchvision, Pillow and numpy, not ttnn, so an encoder
service can resize images in a process that never opens a device.

It gives what the checkpoint's Hugging Face image processor (MiniMaxM3VLImageProcessor, transformers 5.12.1) gives
for one image, bit for bit, in two steps:

  resize_image(image) -> ResizedImage(pixels, grid_h, grid_w)
    1. A PIL image is converted to RGB as the processor converts it (PIL's convert, which drops an alpha channel). A
       numpy array must be uint8 RGB or RGBA, (height, width, 3 or 4); its alpha channel is dropped the same way.
    2. The size (resized_size): each side rounded to a multiple of 28 pixels (2 x 2 patches of 14), then both sides
       scaled down or up at the same aspect ratio if the area is over 672 * 672 or under 56 * 56 pixels. An aspect
       ratio over 200 is rejected.
    3. An antialiased bicubic resize of the uint8 image. On the CPU torchvision keeps a uint8 image uint8 through a
       bicubic resize, so the processor's own resized image is uint8 too and pixels holds exactly its values.
    pixels is the resized image, uint8 (height, width, 3). grid_h and grid_w count its blocks of 28 x 28 pixels, the
    vision tokens down and across it: height / 28 and width / 28.
  patch_rows(pixels) -> (pixel_values, grid_thw)
    4. Rescale to [0, 1] and normalize with the CLIP mean and std, in fp32, with the processor's fused arithmetic.
    5. The image as both frames of a temporal patch, cut into 14 x 14 patches with each 2 x 2 block of patches
       consecutive (the projector merges each block into one vision token): one row of 3 channels x 2 frames x 14 x 14
       values per patch.
    pixel_values (h * w, 1176) float32 and grid_thw [[1, h, w]] int64, in patches (h = height / 14, w = width / 14):
    what the processor returns for the image.

preprocess(image) runs both. Like the processor given a decoded image, neither step applies EXIF orientation: decode
photos with PIL.ImageOps.exif_transpose first, as transformers' load_image does.

This copies the processor's steps rather than calling it for three reasons: the encoder service resizes in one
process and runs the encoder in another, and the processor returns only the normalized rows, not the resized uint8
image to send between them; importing it means importing transformers, seconds more of start-up; and on one core it
took 1.9-4.3x the CPU time of this code for one image of 640x480 to 4032x3024 pixels. Its equality with the
processor is checked by tests/unit/test_vision_preprocess_vs_hf.py, with torch 2.11.0, torchvision 0.26.0 and Pillow
12.3.0, the versions tt-metal pins. The resized pixels are what torchvision's uint8 resize gives, so other versions of
torch or torchvision are not covered by that check.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as tvF

from models.demos.minimax_m3.tt.vision.image_input import MERGE_SIZE

# The image processor's settings, which are also the vision config's patch geometry. TtMiniMaxM3VisionEncoder checks
# them against the model's config when it is built.
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
NUM_CHANNELS = 3
PATCH_DIM = NUM_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE  # 1176 values per patch row
BLOCK = PATCH_SIZE * MERGE_SIZE  # 28: the pixels of one vision token along each side
MIN_PIXELS = 56 * 56
MAX_PIXELS = 672 * 672
MAX_TOKENS = MAX_PIXELS // BLOCK**2  # 576
MAX_ASPECT_RATIO = 200
# The most stretched resized image, in tokens along its long side per token along its short one. The rounding of the
# sides can stretch an image beyond its own aspect ratio: 9600 x 48 pixels (width x height; 200:1, the most the
# processor takes) become 9492 x 28 (339:1). With a short side of one token the long side is at most
# floor(sqrt(MAX_PIXELS * 200) / 28) = 339 tokens, and with two it is at most 288.
MAX_RESIZED_ASPECT_RATIO = 339
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

# The processor folds the 1/255 rescale into the mean and std, in fp32; the same tensors give the same results.
_RESCALE_FACTOR = 1 / 255
_MEAN_255 = torch.tensor(IMAGE_MEAN) * (1.0 / _RESCALE_FACTOR)
_STD_255 = torch.tensor(IMAGE_STD) * (1.0 / _RESCALE_FACTOR)


class ResizedImage(NamedTuple):
    """An image resized for the vision encoder. Its fields are those of the encoder service's M3Image."""

    pixels: np.ndarray  # uint8 (height, width, 3), C-contiguous; height and width are multiples of 28
    grid_h: int  # vision tokens down the image: height / 28
    grid_w: int  # vision tokens across the image: width / 28


def resized_size(height: int, width: int) -> Tuple[int, int]:
    """The processor's resize target (its smart_resize) for an image of height x width pixels, (height, width): sides
    rounded to multiples of 28, and scaled down or up, with the aspect ratio kept, when the area is over MAX_PIXELS or
    under MIN_PIXELS.

    Raises ValueError for a size that is not positive and, as the processor does, for an aspect ratio over 200.
    """
    if height < 1 or width < 1:
        raise ValueError(f"image size must be positive, got height {height} and width {width}")
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise ValueError(
            f"image aspect ratio must be at most {MAX_ASPECT_RATIO}, got height {height} and width {width}"
        )
    resized_height = max(BLOCK, round(height / BLOCK) * BLOCK)
    resized_width = max(BLOCK, round(width / BLOCK) * BLOCK)
    if resized_height * resized_width > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        resized_height = math.floor(height / beta / BLOCK) * BLOCK
        resized_width = math.floor(width / beta / BLOCK) * BLOCK
    elif resized_height * resized_width < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        resized_height = math.ceil(height * beta / BLOCK) * BLOCK
        resized_width = math.ceil(width * beta / BLOCK) * BLOCK
    return resized_height, resized_width


def _rgb_chw(image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
    """The image as the processor takes a PIL image: uint8 (3, height, width), a channels-last view of new memory."""
    if isinstance(image, Image.Image):
        # pil_to_tensor copies the pixels and returns a channels-last view, as the processor's input is.
        return tvF.pil_to_tensor(image if image.mode == "RGB" else image.convert("RGB"))
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise TypeError(f"an image array must be uint8, got {image.dtype}")
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"an image array must be RGB or RGBA, (height, width, 3 or 4), got {image.shape}")
        # A copy of the RGB channels, in the layout pil_to_tensor gives.
        return torch.from_numpy(np.array(image[:, :, :3], order="C")).permute(2, 0, 1)
    raise TypeError(f"image must be a PIL image or a uint8 numpy array, got {type(image).__name__}")


def resize_image(image: Union[Image.Image, np.ndarray]) -> ResizedImage:
    """The image resized as the processor resizes it (steps 1-3 of the module docstring), and its token grid.

    image: a PIL image of any mode, or a uint8 numpy array (height, width, 3 or 4) of RGB or RGBA values. Returns new
    memory, never a view of the caller's array. Raises TypeError for another type or dtype and ValueError for another
    shape and for the sizes resized_size rejects.
    """
    rgb = _rgb_chw(image)
    resized_height, resized_width = resized_size(rgb.shape[1], rgb.shape[2])
    # The processor's resize, of a batch of one image. tvF.resize returns its input when the size does not change.
    resized = tvF.resize(
        rgb.unsqueeze(0),
        [resized_height, resized_width],
        interpolation=tvF.InterpolationMode.BICUBIC,
        antialias=True,
    )
    pixels = resized[0].permute(1, 2, 0).contiguous().numpy()
    return ResizedImage(pixels, resized_height // BLOCK, resized_width // BLOCK)


def check_pixels(pixels: np.ndarray) -> Tuple[int, int]:
    """The token grid (grid_h, grid_w) of a resized image, after checking that it is one resize_image can return.

    Raises TypeError unless pixels is a uint8 numpy array, and ValueError unless it is (height, width, 3) with height
    and width positive multiples of 28, an area of 4 to 576 tokens (56 x 56 to 672 x 672 pixels) and an aspect ratio
    of at most MAX_RESIZED_ASPECT_RATIO: every image resize_image returns passes, and nothing else the vision encoder
    has not been built for does.
    """
    if not isinstance(pixels, np.ndarray):
        raise TypeError(f"pixels must be a numpy array, got {type(pixels).__name__}")
    if pixels.dtype != np.uint8:
        raise TypeError(f"pixels must be uint8, got {pixels.dtype}")
    if pixels.ndim != 3 or pixels.shape[2] != NUM_CHANNELS:
        raise ValueError(f"pixels must be an RGB image (height, width, 3), got shape {pixels.shape}")
    height, width = pixels.shape[:2]
    if height < BLOCK or width < BLOCK or height % BLOCK or width % BLOCK:
        raise ValueError(
            f"pixels height and width must be positive multiples of {BLOCK}, got height {height} and width {width}"
        )
    grid_h, grid_w = height // BLOCK, width // BLOCK
    if not MIN_PIXELS <= height * width <= MAX_PIXELS:
        raise ValueError(
            f"pixels must have {MIN_PIXELS // BLOCK**2} to {MAX_TOKENS} vision tokens of {BLOCK} x {BLOCK} pixels,"
            f" got {grid_h * grid_w} ({grid_h} down, {grid_w} across) for height {height} and width {width}"
        )
    if max(grid_h, grid_w) > MAX_RESIZED_ASPECT_RATIO * min(grid_h, grid_w):
        raise ValueError(
            f"pixels aspect ratio must be at most {MAX_RESIZED_ASPECT_RATIO}, the most the resize gives,"
            f" got height {height} and width {width}"
        )
    return grid_h, grid_w


def patch_rows(pixels: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """The patch rows of a resized image (steps 4 and 5 of the module docstring): (pixel_values, grid_thw), what the
    processor returns for the image. pixel_values is (h * w, 1176) float32 and grid_thw the (1, 3) int64 grid
    [[1, h, w]], in patches of 14 x 14 pixels: twice grid_h and grid_w.

    pixels: as resize_image returns it; raises check_pixels' errors before any other work. A read-only array is
    copied first, which torch needs to take it without a warning.
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


def preprocess(image: Union[Image.Image, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """patch_rows(resize_image(image).pixels): what the processor returns for the image, (pixel_values, grid_thw)."""
    return patch_rows(resize_image(image).pixels)

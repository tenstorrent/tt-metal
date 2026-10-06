# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The host side of an image request between the protocol and the model: the image bytes decoded, the checkpoint's
processor geometry (``smart_resize``: each side a multiple of 32 px within the pixel bounds) applied per part with
its ``detail``, the grid and merged-token count per image, the request's image digest, and the pixel patches for
the tower (the tower lane's ``vision_reference.preprocess_image`` when the tree carries it).

The geometry is the checkpoint's ``preprocessor_config.json`` (``Qwen2VLImageProcessorFast``: patch 16, merge 2,
``shortest_edge`` 65,536 and ``longest_edge`` 16,777,216 pixels); ``detail: low`` lowers the pixel cap to 512 x 512
(the OpenAI convention), so a low-detail image is at most 256 tokens.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Any, Sequence

from models.demos.blackhole.qwen38_flash_next.mrope import Qwen38ImageGrid

PATCH_SIZE = 16
MERGE_SIZE = 2
RESIZE_FACTOR = PATCH_SIZE * MERGE_SIZE  # 32: every side of the resized image is a multiple of it
MIN_PIXELS = 65_536  # the processor's shortest_edge: 64 tokens
MAX_PIXELS = 16_777_216  # the processor's longest_edge: 16,384 tokens
LOW_DETAIL_MAX_PIXELS = 512 * 512  # detail "low": at most 256 tokens
MAX_ASPECT_RATIO = 200
MAX_SOURCE_PIXELS = 100_000_000  # the decoded source image (before the processor's resize): 100 MP
DETAIL_MAX_PIXELS = {"auto": MAX_PIXELS, "high": MAX_PIXELS, "low": LOW_DETAIL_MAX_PIXELS}


class Qwen38ImageError(ValueError):
    """An image the server cannot take: undecodable bytes, an aspect ratio over 200, an unknown detail."""


def smart_resize(
    height: int, width: int, *, factor: int = RESIZE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """The processor's resize target: both sides rounded to a multiple of ``factor``, the pixel count scaled into
    ``[min_pixels, max_pixels]`` (``Qwen2VLImageProcessor.smart_resize``)."""

    if height <= 0 or width <= 0:
        raise Qwen38ImageError(f"image sides must be positive, got {height}x{width}")
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise Qwen38ImageError(
            f"image aspect ratio {max(height, width) / min(height, width):.1f} is over {MAX_ASPECT_RATIO}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_grid(height: int, width: int, detail: str = "auto") -> Qwen38ImageGrid:
    """The (1, h, w) patch grid of an image of ``height`` x ``width`` pixels under ``detail``."""

    if detail not in DETAIL_MAX_PIXELS:
        raise Qwen38ImageError(f"detail must be one of {tuple(DETAIL_MAX_PIXELS)}, got {detail!r}")
    resized_height, resized_width = smart_resize(height, width, max_pixels=DETAIL_MAX_PIXELS[detail])
    return Qwen38ImageGrid(1, resized_height // PATCH_SIZE, resized_width // PATCH_SIZE)


@dataclass(frozen=True)
class Qwen38DecodedImage:
    """One decoded image: RGB pixels (a PIL image), its size, grid, digest and detail."""

    image: Any
    width: int
    height: int
    detail: str
    grid: Qwen38ImageGrid
    sha256: str

    @property
    def merged_tokens(self) -> int:
        return self.grid.merged_tokens


def decode_image(data: bytes, detail: str = "auto") -> Qwen38DecodedImage:
    """Decode image bytes (JPEG, PNG, WEBP, GIF's first frame, BMP) to RGB and derive the processor grid."""

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as opened:
            width, height = opened.size  # the header alone: refuse a bomb before its pixels are allocated
            if width * height > MAX_SOURCE_PIXELS:
                raise Qwen38ImageError(
                    f"image of {width}x{height} px is over the {MAX_SOURCE_PIXELS} pixel source limit"
                )
            smart_resize(height, width, max_pixels=DETAIL_MAX_PIXELS.get(detail, MAX_PIXELS))  # the aspect rule too
            opened.load()
            image = opened.convert("RGB")
    except Qwen38ImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise Qwen38ImageError(f"the image bytes could not be decoded: {error}") from error
    width, height = image.size
    grid = image_grid(height, width, detail)
    digest = hashlib.sha256(data).hexdigest()
    return Qwen38DecodedImage(image, width, height, detail, grid, digest)


def image_digests(images: Sequence[Qwen38DecodedImage]) -> tuple[str, ...]:
    """The per-image keys in prompt order (the bytes' digest, the detail, the grid): two images of one size render
    identical pad ids, so the key beside the ids is what tells a committed prefix's image from another."""

    return tuple(f"{image.sha256}:{image.detail}:{image.grid.as_tuple()}" for image in images)


def request_digest(images: Sequence[Qwen38DecodedImage]) -> str:
    """One digest over a request's images in prompt order (the request's record; the reuse key is per image)."""

    digest = hashlib.sha256()
    for key in image_digests(images):
        digest.update(f"{key};".encode())
    return digest.hexdigest()


def pixel_patches(image: Qwen38DecodedImage):
    """The tower's input for one image: ``[N, 1536]`` fp32 pixel patches in the processor's order (channel, temporal
    copy, 16 x 16), through the tower lane's ``vision_reference.preprocess_image`` (the processor's bicubic resize
    and normalisation, matched to it on the pinned fixtures).  Raises when the tree has no tower module."""

    try:
        from models.demos.blackhole.qwen38_flash_next import vision_reference
    except ImportError as error:  # the tower lane's module lands beside this one
        raise Qwen38ImageError("this build carries no vision tower preprocessing (vision_reference)") from error
    import dataclasses

    import numpy as np
    import torch

    pixels = torch.from_numpy(np.asarray(image.image)).permute(2, 0, 1).contiguous()  # [3, H, W] uint8
    # The tower's preprocessing under this part's detail: the pixel cap the grid was computed with.
    config = dataclasses.replace(vision_reference.VisionTowerConfig(), max_pixels=DETAIL_MAX_PIXELS[image.detail])
    patches, grid_thw = vision_reference.preprocess_image(pixels, config)
    if tuple(int(v) for v in grid_thw.reshape(-1).tolist()) != image.grid.as_tuple():
        raise Qwen38ImageError(
            f"the tower's grid {grid_thw.tolist()} differs from the prompt's {image.grid.as_tuple()}"
        )
    return patches

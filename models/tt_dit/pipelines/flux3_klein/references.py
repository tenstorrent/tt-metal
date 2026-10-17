# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein reference-image sizing on the host: BFL's ``pipeline.py:129-180`` and the ``generate.py:65-77``
glue, expression for expression (design record ``docs/M8_DESIGN.md`` sections 1.1 and 3).

Everything here is part of the checkpoint's numerical contract: the canvas a reference is fitted to decides the token
grid the DiT sees, the LANCZOS resample and the reflect pad decide the pixels the VAE encodes, and the crop box decides
the pixels the user gets back. So nothing may be reassociated: Python's banker's ``round``, ``math.sqrt``, the ``0.995``
retry loop, ``int()`` floors, ``np.pad`` modes and the PIL resample filter are the reference's own. The reference
package is never imported here; the host test holds every function bit-exact against the vendored copy.

Rules (``pipeline.py:24-28``):
  * one reference: ``reference_canvas`` fits it under 4 MiP WITHOUT upscaling (``min(1.0, ...)``) and rounds the fitted
    size up to the 16-px grid, clamped to [64, 4096] per side; the image is centred on that canvas with a reflect pad
    (``symmetric`` when a side is <= 16 px) and the fitted box is returned so the render can be cropped back to it;
  * several references: each is fitted under 1 MiP (sides floored) and centre-CROPPED to the 16-px grid, no padding,
    no crop-back;
  * the default canvas is the first PREPARED reference's size (1024 x 1024 without references);
  * the crop-back applies only to a single reference AND only when the caller gave neither width nor height.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .packing import FLUX3_KLEIN_DOWNSAMPLING_RATIO, FLUX3_KLEIN_MAX_REFERENCES, reference_ids

if TYPE_CHECKING:
    from PIL import Image

# reference pipeline.py:25: the first reference, padded canvas included.
FLUX3_KLEIN_MATCH_MAX_PIXELS = 4 * 1024 * 1024
# reference pipeline.py:26
FLUX3_KLEIN_MIN_SIDE, FLUX3_KLEIN_MAX_SIDE = 64, 4096
# reference pipeline.py:28: each reference of a multi-reference edit.
FLUX3_KLEIN_MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024
# generate.py:36-37 / pipeline.py:64-65
TOO_MANY_REFERENCES = f"at most {FLUX3_KLEIN_MAX_REFERENCES} references"

CropBox = tuple[int, int, int, int]
Size = tuple[int, int]


def ceil16(v: int) -> int:
    """``_ceil16`` (pipeline.py:129-130): ``v`` rounded up to the 16-px grid."""
    return -(-v // FLUX3_KLEIN_DOWNSAMPLING_RATIO) * FLUX3_KLEIN_DOWNSAMPLING_RATIO


def reference_canvas(width: int, height: int) -> tuple[Size, Size]:
    """(fitted size, canvas) of a single reference (pipeline.py:133-143): the canvas is the fitted size rounded up to
    16 and clamped to [64, 4096] per side, at most 4 MiP; the scale never exceeds 1 (no upscaling)."""
    scale = min(
        1.0,
        math.sqrt(FLUX3_KLEIN_MATCH_MAX_PIXELS / (width * height)),
        FLUX3_KLEIN_MAX_SIDE / max(width, height),
    )
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(FLUX3_KLEIN_MAX_SIDE, max(FLUX3_KLEIN_MIN_SIDE, ceil16(v))) for v in fit)
        if canvas[0] * canvas[1] <= FLUX3_KLEIN_MATCH_MAX_PIXELS:
            return fit, canvas
        scale *= 0.995


def prepare_reference(image: Image.Image) -> tuple[Image.Image, CropBox | None]:
    """A single reference (pipeline.py:145-158): RGB, fitted by :func:`reference_canvas` and reflect-padded to its
    canvas, centred. Returns the image and, for a padded one, the (left, top, right, bottom) box of the fitted image."""
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (w, h) = reference_canvas(image.width, image.height)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (w, h) == image.size:
        return image, None
    left, top = (w - image.width) // 2, (h - image.height) // 2
    padding = ((top, h - image.height - top), (left, w - image.width - left), (0, 0))
    # reflect needs a pad shorter than the side
    mode = "reflect" if min(image.size) > FLUX3_KLEIN_DOWNSAMPLING_RATIO else "symmetric"
    padded = Image.fromarray(np.pad(np.asarray(image), padding, mode=mode))
    return padded, (left, top, left + image.width, top + image.height)


def prepare_multi_reference(image: Image.Image) -> Image.Image:
    """Each reference of a multi-reference edit (pipeline.py:160-171): RGB, fitted under 1 MiP (sides floored),
    centre-cropped to the 16-px grid."""
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > FLUX3_KLEIN_MULTI_REFERENCE_MAX_PIXELS:
        scale = math.sqrt(FLUX3_KLEIN_MULTI_REFERENCE_MAX_PIXELS / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    r = FLUX3_KLEIN_DOWNSAMPLING_RATIO
    w, h = width // r * r, height // r * r
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h))


def default_size(references: Sequence[Image.Image]) -> Size:
    """1024 x 1024 for text-to-image; the first (prepared) reference's own size otherwise (pipeline.py:174-176)."""
    return tuple(references[0].size) if references else (1024, 1024)


def to_tensor(image: Image.Image) -> torch.Tensor:
    """pipeline.py:179-180: fp32 ``[3, H, W]`` in [-1, 1] (``uint8 / 127.5 - 1``)."""
    return torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 127.5 - 1.0


def open_image(source) -> Image.Image:
    """A ``PIL.Image.Image`` as is, else ``Image.open(source)`` (a path or a file object)."""
    from PIL import Image

    return source if isinstance(source, Image.Image) else Image.open(source)


@dataclass(frozen=True)
class PreparedReferences:
    """The references after :func:`prepare_references`: what the VAE encoder gets and what the canvas defaults to."""

    images: tuple
    """Prepared RGB ``PIL.Image.Image`` s, sizes multiples of 16."""
    crop: CropBox | None
    """The single-reference crop-back box (applied to the decoded PIL image), ``None`` otherwise."""
    sizes: tuple[Size, ...]
    """``(width, height)`` per prepared image."""
    default_size: Size
    """``default_size(images)``."""

    def latent_grids(self) -> tuple[tuple[int, int], ...]:
        """``(h, w)`` token grid per reference (``H // 16``, ``W // 16``)."""
        r = FLUX3_KLEIN_DOWNSAMPLING_RATIO
        return tuple((height // r, width // r) for width, height in self.sizes)

    def cond_tokens(self) -> int:
        """Total ``image_cond`` rows the DiT sees for these references."""
        return sum(h * w for h, w in self.latent_grids())

    def encoder_levels(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Per reference, the ``(rows, channels)`` of the four encoder levels (full, /2, /4, /8 resolution; the strided
        downsample gives ``floor(H / 2)``, ``autoencoder.py:34-40``; even on every 16-multiple side)."""
        out = []
        for width, height in self.sizes:
            levels = []
            h, w = height, width
            for channels in (128, 256, 512, 512):
                levels.append((h * w, channels))
                h, w = (h + 1 - 3) // 2 + 1, (w + 1 - 3) // 2 + 1
            out.append(tuple(levels))
        return tuple(out)


def prepare_references(images: Sequence, *, width: int | None, height: int | None) -> PreparedReferences:
    """``generate.py:65-72``: several images -> :func:`prepare_multi_reference` each and no crop; one image ->
    :func:`prepare_reference`, its crop box kept only when neither ``width`` nor ``height`` was given. More than
    ``FLUX3_KLEIN_MAX_REFERENCES`` images raise ``ValueError`` BEFORE any PIL work (generate.py:36-37). ``images`` are
    ``PIL.Image.Image`` s or anything ``Image.open`` accepts."""
    if len(images) > FLUX3_KLEIN_MAX_REFERENCES:
        raise ValueError(TOO_MANY_REFERENCES)
    opened = [open_image(source) for source in images]
    if len(opened) > 1:
        references, crop = [prepare_multi_reference(image) for image in opened], None
    else:
        prepared = [prepare_reference(image) for image in opened]
        references = [image for image, _ in prepared]
        crop = prepared[0][1] if prepared and not (width or height) else None
    return PreparedReferences(
        images=tuple(references),
        crop=crop,
        sizes=tuple(tuple(image.size) for image in references),
        default_size=default_size(references),
    )


def reference_ids_for(prepared: PreparedReferences) -> torch.Tensor:
    """fp32 ``[N, 4]``: ``packing.reference_ids(h_i, w_i, i)`` of every reference concatenated in reference order
    (``pipeline.py:84-95`` ``encode_references``: scaled ids with ``l = -(1 + index)``)."""
    ids = [reference_ids(h, w, index) for index, (h, w) in enumerate(prepared.latent_grids())]
    return torch.cat(ids) if ids else torch.zeros((0, 4), dtype=torch.float32)


__all__ = [
    "FLUX3_KLEIN_MATCH_MAX_PIXELS",
    "FLUX3_KLEIN_MAX_SIDE",
    "FLUX3_KLEIN_MIN_SIDE",
    "FLUX3_KLEIN_MULTI_REFERENCE_MAX_PIXELS",
    "TOO_MANY_REFERENCES",
    "CropBox",
    "PreparedReferences",
    "ceil16",
    "default_size",
    "open_image",
    "prepare_multi_reference",
    "prepare_reference",
    "prepare_references",
    "reference_canvas",
    "reference_ids_for",
    "to_tensor",
]

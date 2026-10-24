# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein reference-image sizing and pixel conversion on the host: ``Flux2KleinPipeline.__call__`` step 4
(``pipeline_flux2_klein.py:764-787``, cited as ``P``) and the ``Flux2ImageProcessor`` / ``VaeImageProcessor``
methods it calls (``diffusers/pipelines/flux2/image_processor.py`` ``FIP``, ``diffusers/image_processor.py``
``VIP``), expression for expression; BFL ``sampling.py:52-67, 226-237`` is the second citation (its 2024² single-
reference cap is NOT the diffusers rule and is not mirrored).

Everything here is part of the checkpoint's numerical contract: the size a reference is cut to decides the token
grid the DiT sees, the LANCZOS resample and the centre crop decide the pixels the VAE encodes. So nothing may be
reassociated: ``math.sqrt``, ``int()`` floors, the ``// 2`` crop origin, the PIL resample filter and the fp32
``/ 255`` -> ``2x - 1`` chain are the reference's own. diffusers is never imported here.

Rules, per condition image (``P:769-783``):
  * ``check_image_input``: both sides >= 64 px and aspect ratio <= 8 (``FIP:67-105``), else ``ValueError``;
  * over 1 MiP: ``_resize_to_target_area(img, 1024 * 1024)`` -- ``scale = sqrt(area / (w * h))``, sides ``int(w *
    scale)`` / ``int(h * scale)``, LANCZOS, in the image's own mode (``FIP:107-115``);
  * sides floored to the 16-px grid (``P:778-780``), ``preprocess(resize_mode="crop")``: a centre crop with origin
    ``(w - W) // 2`` (``FIP:125-153``), then ``convert("RGB")`` (``VIP:237-251``, ``do_convert_rgb`` is true for
    ``Flux2ImageProcessor``), then ``np.array(img).astype(float32) / 255.0`` and ``2.0 * x - 1.0`` (``VIP:152-168``,
    ``VIP:207-220``);
  * the default canvas is the FIRST condition image's floored size (``height = height or image_height``,
    ``P:782-783``), 1024 x 1024 without references (``P:785-787``, ``default_sample_size 128 * 8``).

Host cost: ``Image.open`` is lazy, so the PNG decode (40-100 ms per 1 MiP reference) happens inside ``convert("RGB")``
of :func:`prepare_reference`, and :func:`to_pixels` (the fp32 ``/ 255`` -> ``2x - 1`` chain in the encoder's channels-
last upload layout, ~10 ms) follows; both run on every request. :class:`ReferenceCache` (one per pipeline) keeps the
prepared image and its pixels of a PATH source keyed by ``(realpath, st_mtime_ns, st_size)`` (:func:`reference_key`),
so a reference served again, as the benchmark matrix's and an editing session's are, costs a stat. The cached bytes
are the ones the chain above produced (nothing is recomputed differently), so the reference tokens are bit-equal; a
``PIL.Image.Image`` or file object has no key and is prepared per request as before.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from .packing import FLUX2_KLEIN_DOWNSAMPLING_RATIO, FLUX2_KLEIN_MAX_REFERENCES, reference_ids

if TYPE_CHECKING:
    from PIL import Image

# The canvas the DiT serves: every side a multiple of 16 (P:778-780, P:493-494) up to the port's structural limit.
FLUX2_KLEIN_MIN_SIDE, FLUX2_KLEIN_MAX_SIDE = 16, 4096
# FIP:67-70 check_image_input defaults: min_side_length 64, max_aspect_ratio 8.
FLUX2_KLEIN_REFERENCE_MIN_SIDE = 64
FLUX2_KLEIN_REFERENCE_MAX_ASPECT = 8
# P:774-775: a condition image above this many pixels is resized onto it.
FLUX2_KLEIN_REFERENCE_MAX_PIXELS = 1024 * 1024
# P:785-787: default_sample_size * vae_scale_factor = 128 * 8.
FLUX2_KLEIN_DEFAULT_SIZE = (1024, 1024)
TOO_MANY_REFERENCES = f"at most {FLUX2_KLEIN_MAX_REFERENCES} references"
# :class:`ReferenceCache` entries: a 1 MiP reference holds ~3 MB of prepared RGB + 12 MB of fp32 pixels (~250 MB bound).
FLUX2_KLEIN_REFERENCE_CACHE_SIZE = 16

Size = tuple[int, int]
ReferenceKey = tuple[str, int, int]


def check_image_input(image: Image.Image) -> Image.Image:
    """``FIP:67-105`` ``check_image_input``: both sides at least 64 px and an aspect ratio of at most 8:1."""
    width, height = image.size
    if width < FLUX2_KLEIN_REFERENCE_MIN_SIDE or height < FLUX2_KLEIN_REFERENCE_MIN_SIDE:
        raise ValueError(
            f"Image too small: {width}x{height}. Both dimensions must be at least {FLUX2_KLEIN_REFERENCE_MIN_SIDE}px"
        )
    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > FLUX2_KLEIN_REFERENCE_MAX_ASPECT:
        raise ValueError(
            f"Aspect ratio too extreme: {width}x{height} (ratio: {aspect_ratio:.1f}:1). "
            f"Maximum allowed ratio is {FLUX2_KLEIN_REFERENCE_MAX_ASPECT}:1"
        )
    return image


def resize_to_target_area(image: Image.Image, target_area: int = FLUX2_KLEIN_REFERENCE_MAX_PIXELS) -> Image.Image:
    """``FIP:107-115`` ``_resize_to_target_area``: scale both sides by ``sqrt(target_area / (w * h))``, floor with
    ``int()``, LANCZOS."""
    from PIL import Image

    image_width, image_height = image.size
    scale = math.sqrt(target_area / (image_width * image_height))
    width = int(image_width * scale)
    height = int(image_height * scale)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def floor16(v: int) -> int:
    """``P:777-780``: ``(v // multiple_of) * multiple_of`` with ``multiple_of = vae_scale_factor * 2 = 16``."""
    return (v // FLUX2_KLEIN_DOWNSAMPLING_RATIO) * FLUX2_KLEIN_DOWNSAMPLING_RATIO


def center_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    """``FIP:125-153`` ``_resize_and_crop`` (``resize_mode="crop"``): a centre crop, origin ``(w - width) // 2``,
    ``(h - height) // 2``; no resampling."""
    image_width, image_height = image.size
    left = (image_width - width) // 2
    top = (image_height - height) // 2
    return image.crop((left, top, left + width, top + height))


def prepare_reference(image: Image.Image) -> Image.Image:
    """One condition image (``P:769-783``): :func:`check_image_input`, the 1 MiP cap, the 16-px floor, the centre
    crop (``preprocess`` with ``height`` / ``width`` already on the grid: ``get_default_height_width`` keeps them,
    ``VIP:600-602``), and the RGB conversion ``preprocess`` applies after the resize (``VIP:697-698``). The result is
    what ``pil_to_numpy`` sees."""
    check_image_input(image)
    image_width, image_height = image.size
    if image_width * image_height > FLUX2_KLEIN_REFERENCE_MAX_PIXELS:
        image = resize_to_target_area(image, FLUX2_KLEIN_REFERENCE_MAX_PIXELS)
        image_width, image_height = image.size
    image = center_crop(image, floor16(image_width), floor16(image_height))
    return image.convert("RGB")


def to_tensor(image: Image.Image) -> torch.Tensor:
    """``pil_to_numpy`` -> ``numpy_to_pt`` -> ``normalize`` (``VIP:152-168``, ``VIP:172-189``, ``VIP:207-220``) for one
    prepared RGB image: fp32 ``[3, H, W]`` in [-1, 1] as ``2.0 * (uint8 / 255.0) - 1.0``."""
    array = np.array(image).astype(np.float32) / 255.0
    x = torch.from_numpy(array.transpose(2, 0, 1))
    return 2.0 * x - 1.0


def to_pixels(image: Image.Image) -> torch.Tensor:
    """:func:`to_tensor` in the VAE encoder's upload layout: the same fp32 values as a contiguous ``[1, H, W, 3]``
    (``P:524``: the condition image enters the VAE channels-last on the mesh)."""
    return to_tensor(image).permute(1, 2, 0)[None].contiguous()


def image_to_uint8(image: torch.Tensor) -> torch.Tensor:
    """``postprocess(output_type="pil")`` arithmetic (``VIP:738-787``) for one decoded ``[3, H, W]`` image:
    ``denormalize`` ``(x * 0.5 + 0.5).clamp(0, 1)`` (``VIP:222-235``), ``pt_to_numpy`` to fp32 ``[H, W, 3]``
    (``VIP:191-205``), ``numpy_to_pil`` ``(x * 255).round()`` to uint8 (``VIP:128-150``): ROUNDED, not truncated."""
    x = (image * 0.5 + 0.5).clamp(0, 1)
    x = x.permute(1, 2, 0).float()
    return (x * 255).round().to(torch.uint8)


def open_image(source) -> Image.Image:
    """A ``PIL.Image.Image`` as is, else ``Image.open(source)`` (a path or a file object)."""
    from PIL import Image

    return source if isinstance(source, Image.Image) else Image.open(source)


def reference_key(source) -> ReferenceKey | None:
    """``(realpath, st_mtime_ns, st_size)`` of a path-like source: what :class:`ReferenceCache` keys on (the same file
    at the same version). ``None`` for a ``PIL.Image.Image`` or a file object: prepared per request."""
    if isinstance(source, (str, bytes, os.PathLike)):
        st = os.stat(source)
        return (os.path.realpath(os.fsdecode(source)), int(st.st_mtime_ns), int(st.st_size))
    return None


@dataclass(frozen=True, eq=False)
class PreparedReference:
    """One source through :func:`prepare_reference` with its :func:`to_pixels` (what the VAE encoder uploads)."""

    image: object
    """The prepared RGB ``PIL.Image.Image`` (sides multiples of 16)."""
    pixels: torch.Tensor
    """``to_pixels(image)``: fp32 ``[1, H, W, 3]`` in [-1, 1]."""

    @classmethod
    def of(cls, source) -> PreparedReference:
        image = prepare_reference(open_image(source))
        return cls(image=image, pixels=to_pixels(image))


class ReferenceCache:
    """Prepared references of PATH sources, least recently used out, ``maxsize`` entries (the pipeline holds one). A
    source without a :func:`reference_key` is prepared and returned uncached. ``hits`` / ``misses`` count the keyed
    lookups."""

    def __init__(self, maxsize: int = FLUX2_KLEIN_REFERENCE_CACHE_SIZE) -> None:
        if int(maxsize) < 1:
            raise ValueError(f"maxsize must be at least 1, got {maxsize}")
        self.maxsize = int(maxsize)
        self._entries: OrderedDict[ReferenceKey, PreparedReference] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        # An EMPTY cache is still a cache: ``if pipeline.reference_cache:`` must not read as "caching off".
        return True

    def __contains__(self, source) -> bool:
        key = reference_key(source)
        return key is not None and key in self._entries

    def prepare(self, source) -> PreparedReference:
        key = reference_key(source)
        if key is None:
            return PreparedReference.of(source)
        entry = self._entries.get(key)
        if entry is not None:
            self.hits += 1
            self._entries.move_to_end(key)
            return entry
        self.misses += 1
        entry = PreparedReference.of(source)
        self._entries[key] = entry
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
        return entry

    def clear(self) -> None:
        self._entries.clear()

    def info(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "currsize": len(self._entries), "maxsize": self.maxsize}


def default_size(references: Sequence[Image.Image], *, width: int | None = None, height: int | None = None) -> Size:
    """The canvas ``P:782-787`` resolves: the caller's side where given (``height or ...`` keeps a non-zero value),
    else the first PREPARED reference's own side, else 1024."""
    first_width, first_height = references[0].size if references else (None, None)
    return (
        width or first_width or FLUX2_KLEIN_DEFAULT_SIZE[0],
        height or first_height or FLUX2_KLEIN_DEFAULT_SIZE[1],
    )


@dataclass(frozen=True)
class PreparedReferences:
    """The references after :func:`prepare_references`: what the VAE encoder gets and what the canvas defaults to."""

    images: tuple
    """Prepared RGB ``PIL.Image.Image`` s, sizes multiples of 16."""
    sizes: tuple[Size, ...]
    """``(width, height)`` per prepared image."""
    default_size: Size
    """:func:`default_size` of the images under the request's ``width`` / ``height``."""
    pixels: tuple = field(default=(), compare=False, repr=False)
    """:func:`to_pixels` per prepared image (fp32 ``[1, H, W, 3]``), what the VAE encoder uploads; empty for an object
    built by hand (:meth:`pixel_tensor` then computes them)."""

    def pixel_tensor(self, index: int) -> torch.Tensor:
        """``to_pixels(images[index])``, from :attr:`pixels` when held."""
        return self.pixels[index] if self.pixels else to_pixels(self.images[index])

    def latent_grids(self) -> tuple[tuple[int, int], ...]:
        """``(h, w)`` token grid per reference (``H // 16``, ``W // 16``; ``P:466-467`` encode + patchify)."""
        r = FLUX2_KLEIN_DOWNSAMPLING_RATIO
        return tuple((height // r, width // r) for width, height in self.sizes)

    def cond_tokens(self) -> int:
        """Total reference rows the DiT sees (``P:537-539``: every packed reference concatenated)."""
        return sum(h * w for h, w in self.latent_grids())


def prepare_references(
    images: Sequence, *, width: int | None, height: int | None, cache: ReferenceCache | None = None
) -> PreparedReferences:
    """``P:764-787``: every condition image through :func:`prepare_reference` in order (with its :func:`to_pixels`),
    the default canvas from :func:`default_size`. More than ``FLUX2_KLEIN_MAX_REFERENCES`` images raise ``ValueError``
    BEFORE any PIL work. ``images`` are ``PIL.Image.Image`` s or anything ``Image.open`` accepts; with ``cache``, a path
    source is served from / stored in it (:class:`ReferenceCache`)."""
    if len(images) > FLUX2_KLEIN_MAX_REFERENCES:
        raise ValueError(TOO_MANY_REFERENCES)
    entries = [cache.prepare(source) if cache is not None else PreparedReference.of(source) for source in images]
    references = [entry.image for entry in entries]
    return PreparedReferences(
        images=tuple(references),
        sizes=tuple(tuple(image.size) for image in references),
        default_size=default_size(references, width=width, height=height),
        pixels=tuple(entry.pixels for entry in entries),
    )


def reference_ids_for(prepared: PreparedReferences) -> torch.Tensor:
    """int64 ``[N, 4]``: ``packing.reference_ids(h_i, w_i, i)`` of every reference concatenated in reference order
    (``P:353-365`` ``_prepare_image_ids``: ``t = 10 + 10 i``, then ``torch.cat``)."""
    ids = [reference_ids(h, w, index) for index, (h, w) in enumerate(prepared.latent_grids())]
    return torch.cat(ids) if ids else torch.zeros((0, 4), dtype=torch.int64)


__all__ = [
    "FLUX2_KLEIN_DEFAULT_SIZE",
    "FLUX2_KLEIN_MAX_SIDE",
    "FLUX2_KLEIN_MIN_SIDE",
    "FLUX2_KLEIN_REFERENCE_MAX_ASPECT",
    "FLUX2_KLEIN_REFERENCE_CACHE_SIZE",
    "FLUX2_KLEIN_REFERENCE_MAX_PIXELS",
    "FLUX2_KLEIN_REFERENCE_MIN_SIDE",
    "TOO_MANY_REFERENCES",
    "PreparedReference",
    "PreparedReferences",
    "ReferenceCache",
    "ReferenceKey",
    "Size",
    "center_crop",
    "check_image_input",
    "default_size",
    "floor16",
    "image_to_uint8",
    "open_image",
    "prepare_reference",
    "prepare_references",
    "reference_ids_for",
    "reference_key",
    "resize_to_target_area",
    "to_pixels",
    "to_tensor",
]

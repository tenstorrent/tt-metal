# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 serving policy: the envelope of requests a deployment commits to serving."""

from __future__ import annotations

import os
from collections.abc import Iterator

from PIL import Image

from .packing import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_MAX_ASPECT_RATIO,
    MINIMAX_H3_MAX_PIXELS,
    MINIMAX_H3_MIN_ASPECT_RATIO,
    resolve_canvas_size,
)
from .packing_ref2va import resolve_reference_image_size

MINIMAX_H3_ASPECT_RATIOS = ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16))
MINIMAX_H3_DEFAULT_ASPECT_RATIO = (16, 9)

MINIMAX_H3_DURATIONS_S = tuple(range(4, 16))
MINIMAX_H3_DEFAULT_DURATION_S = 5

# Served denoising step count.
MINIMAX_H3_NUM_INFERENCE_STEPS = 50

# fl2va keyframe rows in the prompt arena, fixed by the checkpoint. Per keyframe: "<Picture i>: "
# label (6 tokens), <|vision_start|>, one <|image_pad|> per merged 32x32 patch, <|vision_end|>.
# The largest canvas resolve_canvas_size can produce is 576x1856 (a 1:4 keyframe snaps above the
# area cap) = 18*58 = 1044 patches -> 1052 rows/keyframe; two keyframes = 2104 -> 2112 tile-aligned.
MINIMAX_H3_MAX_KEYFRAME_TOKENS = 2112

# Longest prompt a t2va/fl2va deployment accepts; the prompt arena cap is derived from it. The
# default 3008 fills a 5120-row prompt arena (5120 - 2112), the arena the shipped bucket ladder was
# sized around; the model itself imposes no prompt limit. Must be a multiple of TILE_SIZE so the
# derived cap stays tile-aligned, and the caps sum must still fit the top rung (3680 with the
# default ladder).
MINIMAX_H3_MAX_TEXT_TOKENS = int(os.environ.get("MINIMAX_H3_MAX_TEXT_TOKENS", 3008))

# Served ref2va image resize mode; other modes are rejected on the request path.
MINIMAX_H3_SERVED_REFERENCE_RESIZE_MODE = "match"

# MiniMax API limits on an fl2va keyframe.
MINIMAX_H3_KEYFRAME_MIN_SIDE = 256
MINIMAX_H3_KEYFRAME_MAX_SIDE = 5760


def minimax_h3_parse_aspect_ratio(value: str) -> tuple[int, int]:
    """`"16:9"` -> `(16, 9)`, restricted to the published set (rejects rather than rounds)."""
    text = str(value).strip().replace("x", ":").replace("/", ":")
    parts = text.split(":")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise ValueError(
            f"aspect_ratio must look like 'W:H' (got {value!r}); supported: "
            + ", ".join(f"{w}:{h}" for w, h in MINIMAX_H3_ASPECT_RATIOS)
        )
    pair = (int(parts[0]), int(parts[1]))
    if pair not in MINIMAX_H3_ASPECT_RATIOS:
        raise ValueError(
            f"aspect_ratio {pair[0]}:{pair[1]} is not served; supported: "
            + ", ".join(f"{w}:{h}" for w, h in MINIMAX_H3_ASPECT_RATIOS)
        )
    return pair


def minimax_h3_frames_are_aligned(num_frames: int) -> bool:
    """`num_frames` must be `17n + 5`: 124, 243, 362, ..."""
    return (
        num_frames >= MINIMAX_H3_LATENTS_PER_CHUNK
        and num_frames % MINIMAX_H3_FRAMES_PER_CHUNK == MINIMAX_H3_LATENTS_PER_CHUNK
    )


def validate_input(
    *,
    image: Image.Image | None,
    last_image: Image.Image | None,
    aspect_ratio: tuple[int, int],
    height: int | None,
    width: int | None,
) -> None:
    """Reject a t2va/fl2va request outside the served envelope; any keyframe makes it fl2va."""
    if (height is None) != (width is None):
        raise ValueError("pass both height and width, or neither")
    if image is None and last_image is None:
        validate_t2va(aspect_ratio=aspect_ratio, height=height, width=width)
    else:
        validate_fl2va(image=image, last_image=last_image, height=height, width=width)


def validate_t2va(*, aspect_ratio: tuple[int, int], height: int | None, width: int | None) -> None:
    """`aspect_ratio` must be a published ratio unless an explicit canvas replaces it."""
    if height is None:
        if aspect_ratio not in MINIMAX_H3_ASPECT_RATIOS:
            raise ValueError(
                f"aspect_ratio {aspect_ratio} is not served for T2VA; supported: "
                + ", ".join(f"{w}:{h}" for w, h in MINIMAX_H3_ASPECT_RATIOS)
            )
    else:
        _validate_explicit_canvas(height, width)


def validate_fl2va(
    *,
    image: Image.Image | None,
    last_image: Image.Image | None,
    height: int | None,
    width: int | None,
) -> None:
    """Every keyframe must be within the side and aspect ratio limits, and an explicit canvas must be servable."""
    for name, keyframe in (("image", image), ("last_image", last_image)):
        if keyframe is None:
            continue
        keyframe_width, keyframe_height = keyframe.size
        if not (
            MINIMAX_H3_KEYFRAME_MIN_SIDE <= min(keyframe_width, keyframe_height)
            and max(keyframe_width, keyframe_height) <= MINIMAX_H3_KEYFRAME_MAX_SIDE
        ):
            raise ValueError(
                f"{name} is {keyframe_width}x{keyframe_height}; each side must be from "
                f"{MINIMAX_H3_KEYFRAME_MIN_SIDE} to {MINIMAX_H3_KEYFRAME_MAX_SIDE} pixels"
            )
        if not MINIMAX_H3_MIN_ASPECT_RATIO <= keyframe_width / keyframe_height <= MINIMAX_H3_MAX_ASPECT_RATIO:
            raise ValueError(f"{name} is {keyframe_width}x{keyframe_height}; its aspect ratio must be from 1:4 to 4:1")
    if height is not None:
        _validate_explicit_canvas(height, width)


def _validate_explicit_canvas(height: int, width: int) -> None:
    # Derived canvases may round slightly above the area cap; an explicit one may not.
    if height % MINIMAX_H3_CANVAS_MULTIPLE or width % MINIMAX_H3_CANVAS_MULTIPLE:
        raise ValueError(f"canvas {height}x{width} must be a multiple of {MINIMAX_H3_CANVAS_MULTIPLE} on both axes")
    if height * width > MINIMAX_H3_MAX_PIXELS:
        raise ValueError(f"canvas {height}x{width} exceeds the {MINIMAX_H3_MAX_PIXELS}-pixel area cap")
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= (width / height) <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(f"canvas {height}x{width} is outside the 1:4 to 4:1 aspect ratio range")


def served_canvases() -> tuple[tuple[int, int], ...]:
    """The unique `(height, width)` canvases the served aspect ratios resolve to, in ratio order."""
    seen: dict[tuple[int, int], None] = {}
    for aspect_width, aspect_height in MINIMAX_H3_ASPECT_RATIOS:
        seen.setdefault(resolve_canvas_size(aspect_width, aspect_height), None)
    return tuple(seen)


def served_reference_image_sizes(target_height: int, target_width: int) -> tuple[tuple[int, int], ...]:
    """One representative match-mode `(height, width)` per distinct vision-run token count."""
    by_tokens: dict[int, tuple[int, int]] = {}
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    large = 4096
    for height in range(multiple, large + 1, multiple):
        for width in range(multiple, large + 1, multiple):
            if width > 4 * height or height > 4 * width:
                continue
            size = resolve_reference_image_size(
                width,
                height,
                mode=MINIMAX_H3_SERVED_REFERENCE_RESIZE_MODE,
                target_width=target_width,
                target_height=target_height,
            )
            by_tokens.setdefault((size[0] // multiple) * (size[1] // multiple), size)
    return tuple(by_tokens[tokens] for tokens in sorted(by_tokens))


def served_reference_canvases() -> tuple[tuple[int, int], ...]:
    """One served canvas per distinct area (same-area canvases produce identical run shapes)."""
    by_area: dict[int, tuple[int, int]] = {}
    for canvas in served_canvases():
        by_area.setdefault(canvas[0] * canvas[1], canvas)
    return tuple(by_area[area] for area in sorted(by_area))


def served_reference_video_canvases() -> tuple[tuple[int, int], ...]:
    """Video-reference canvases whose run token count no image reference reaches."""
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    image_tokens = {
        (height // multiple) * (width // multiple)
        for canvas in served_reference_canvases()
        for height, width in served_reference_image_sizes(*canvas)
    }
    by_tokens: dict[int, tuple[int, int]] = {}
    large = 4096
    for height in range(multiple, large + 1, multiple):
        for width in range(multiple, large + 1, multiple):
            if width > 4 * height or height > 4 * width:
                continue
            canvas = resolve_canvas_size(width, height)
            tokens = (canvas[0] // multiple) * (canvas[1] // multiple)
            if tokens not in image_tokens:
                by_tokens.setdefault(tokens, canvas)
    return tuple(by_tokens[tokens] for tokens in sorted(by_tokens))


def served_envelope(task: str) -> Iterator:
    """The vision-layout warm units a deployment must compile.

    t2va yields `(n_keyframes, canvas)`; ref2va yields `(canvas, image_size)`.
    """
    if task == "t2va":
        yield 0, None
        for canvas in served_canvases():
            yield 1, canvas
            yield 2, canvas
        return
    if task != "ref2va":
        raise NotImplementedError(f"served_envelope is not defined for task {task!r}")
    for canvas in served_reference_canvases():
        for size in served_reference_image_sizes(*canvas):
            yield canvas, size

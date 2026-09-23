# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 serving policy: the envelope of requests a deployment commits to serving."""

from __future__ import annotations

from collections.abc import Iterator

from .packing import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    resolve_canvas_size,
)
from .packing_ref2va import resolve_reference_image_size

MINIMAX_H3_ASPECT_RATIOS = ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16))
MINIMAX_H3_DEFAULT_ASPECT_RATIO = (16, 9)

MINIMAX_H3_DURATIONS_S = tuple(range(4, 16))
MINIMAX_H3_DEFAULT_DURATION_S = 5

# Fixed: the AdaLN modulation table is precomputed per step count.
MINIMAX_H3_NUM_INFERENCE_STEPS = 50

# Text token budget: 5120 prompt arena rows - 2112 for two max-canvas keyframe blocks.
MINIMAX_H3_MAX_TEXT_TOKENS = 3008

# Served ref2va image resize mode; other modes are rejected on the request path.
MINIMAX_H3_SERVED_REFERENCE_RESIZE_MODE = "match"

# ref2va presentation pad targets (multiples of 1024); the top rung is the prompt arena cap.
MINIMAX_H3_REF2VA_PRESENTATION_LADDER = (1024, 4096, 8192, 16384, 32768, 57344)


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

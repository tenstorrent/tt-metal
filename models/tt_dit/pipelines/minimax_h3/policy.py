# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 serving policy: the envelope of requests a deployment commits to serving.

This module holds *choices* -- which aspect ratios, durations and text budget are served -- and
derives their concrete geometry from `packing.py`, which owns the *capability* math
(`resolve_canvas_size`, `align_num_frames`). The dependency is one-way: policy imports packing,
never the reverse.
"""

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

# Not a request lever: the AdaLN modulation table is precomputed per step count, and every served
# shape is warmed at this count.
MINIMAX_H3_NUM_INFERENCE_STEPS = 50

# Token-denominated text budget, enforced by the pipeline itself, not by any client. Sized so two
# max-canvas keyframe vision blocks (2112 tokens) plus a full text budget exactly fill the prompt
# arena cap of 5120 rows: 5120 - 2112 = 3008.
MINIMAX_H3_MAX_TEXT_TOKENS = 3008

# Served ref2va image resize. `max` / `diffusers` stay on the parameter for REPL/tests; the request
# path rejects them so a 2048-short-edge reference cannot escape the warmed envelope.
MINIMAX_H3_SERVED_REFERENCE_RESIZE_MODE = "match"

# Host-side pad targets for a ref2va presentation. Multiples of 1024 so both meshes' SP alignments
# hold; the top rung is the ref2va prompt arena cap. A presentation pads to the next rung that fits
# and raises above the top.
MINIMAX_H3_REF2VA_PRESENTATION_LADDER = (1024, 4096, 8192, 16384, 32768, 57344)


def minimax_h3_parse_aspect_ratio(value: str) -> tuple[int, int]:
    """`"16:9"` -> `(16, 9)`, restricted to the published set.

    Rejects rather than rounds: a caller asking for 2:1 wants 2:1, and quietly serving 16:9 would be
    a wrong answer dressed as a right one.
    """
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
    """`num_frames` must be `17n + 5`: 124, 243, 362, ...

    The modulus lives in `packing` (`FRAMES_PER_CHUNK` / `LATENTS_PER_CHUNK`), so this predicate
    cannot drift from the VAE's chunking.
    """
    return (
        num_frames >= MINIMAX_H3_LATENTS_PER_CHUNK
        and num_frames % MINIMAX_H3_FRAMES_PER_CHUNK == MINIMAX_H3_LATENTS_PER_CHUNK
    )


def served_canvases() -> tuple[tuple[int, int], ...]:
    """The unique `(height, width)` canvases the served aspect ratios resolve to, in ratio order.

    Distinct ratios can resolve to the same canvas after the area cap and the %32 rounding, so the
    result is deduped; the order is the first appearance in `MINIMAX_H3_ASPECT_RATIOS`.
    """
    seen: dict[tuple[int, int], None] = {}
    for aspect_width, aspect_height in MINIMAX_H3_ASPECT_RATIOS:
        seen.setdefault(resolve_canvas_size(aspect_width, aspect_height), None)
    return tuple(seen)


def served_reference_image_sizes(target_height: int, target_width: int) -> tuple[tuple[int, int], ...]:
    """One representative match-mode `(height, width)` per distinct vision-run token count.

    A run's programs are keyed by its token count `(h/32)*(w/32)`, not by the `(h, w)` pair, so two
    aspect splits with equal products are one program; only the product needs a warm unit. The
    resolver is the source of truth, swept over the 1:4..4:1 ratio grid at the target area.
    """
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
    """One served canvas per distinct area: `match` resizes a reference to the target area alone, so
    same-area canvases produce identical run shapes and warming one covers the rest."""
    by_area: dict[int, tuple[int, int]] = {}
    for canvas in served_canvases():
        by_area.setdefault(canvas[0] * canvas[1], canvas)
    return tuple(by_area[area] for area in sorted(by_area))


def served_reference_video_canvases() -> tuple[tuple[int, int], ...]:
    """Video-reference canvases whose run token count no image reference reaches.

    A reference video lands on `resolve_canvas_size` of its OWN aspect, so its canvas family is
    wider than the served set: with the short edge pinned at 768 and the long edge growing before
    the area cap engages, it reaches token counts a `match`-mode image (fixed to the target area)
    never produces. Run programs key on the token count `(h/32)*(w/32)`, so only counts absent from
    the image sweep need their own warm unit.
    """
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

    A `t2va` deployment serves both t2va (the `canvas is None` keyframeless unit) and fl2va (one or
    two keyframes per served canvas), as `(n_keyframes, canvas)`. A `ref2va` deployment yields
    `(canvas, image_size)` for one image per distinct run token count per distinct served area.
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

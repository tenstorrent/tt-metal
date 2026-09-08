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

from .packing import resolve_canvas_size

MINIMAX_H3_ASPECT_RATIOS = ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16))
MINIMAX_H3_DEFAULT_ASPECT_RATIO = (16, 9)

MINIMAX_H3_DURATIONS_S = tuple(range(4, 16))
MINIMAX_H3_DEFAULT_DURATION_S = 5

# Token-denominated text budget, enforced by the pipeline itself, not by any client. Sized so two
# max-canvas keyframe vision blocks (2112 tokens) plus a full text budget exactly fill the prompt
# arena cap of 5120 rows: 5120 - 2112 = 3008.
MINIMAX_H3_MAX_TEXT_TOKENS = 3008


def served_canvases() -> tuple[tuple[int, int], ...]:
    """The unique `(height, width)` canvases the served aspect ratios resolve to, in ratio order.

    Distinct ratios can resolve to the same canvas after the area cap and the %32 rounding, so the
    result is deduped; the order is the first appearance in `MINIMAX_H3_ASPECT_RATIOS`.
    """
    seen: dict[tuple[int, int], None] = {}
    for aspect_width, aspect_height in MINIMAX_H3_ASPECT_RATIOS:
        seen.setdefault(resolve_canvas_size(aspect_width, aspect_height), None)
    return tuple(seen)


def served_envelope(task: str) -> Iterator[tuple[int, tuple[int, int] | None]]:
    """The vision-layout warm units `(n_keyframes, canvas)` a deployment must compile.

    A `t2va` deployment serves both t2va (the `canvas is None` keyframeless unit) and fl2va (one or
    two keyframes per served canvas). ref2va presents references, not keyframes, and is not covered.
    """
    if task != "t2va":
        raise NotImplementedError(f"served_envelope is not defined for task {task!r}")
    yield 0, None
    for canvas in served_canvases():
        yield 1, canvas
        yield 2, canvas

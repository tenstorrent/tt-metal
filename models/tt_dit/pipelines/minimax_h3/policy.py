# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 serving policy: the envelope of requests a deployment commits to serving."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence

import numpy as np
from PIL import Image

from .packing import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_MAX_ASPECT_RATIO,
    MINIMAX_H3_MAX_PIXELS,
    MINIMAX_H3_MIN_ASPECT_RATIO,
    resolve_canvas_size,
)
from .packing_ref2va import (
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    MiniMaxH3Reference,
    resolve_reference_image_size,
    sample_reference_video_frames,
)

MINIMAX_H3_ASPECT_RATIOS = ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16))
MINIMAX_H3_DEFAULT_ASPECT_RATIO = (16, 9)


def align_num_frames(num_frames: int) -> int:
    """Snap a frame count up to the next ``17n + 5`` the video VAE can encode."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    while num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def get_num_frames(duration_s: float) -> int:
    """Frames generated for `duration_s` seconds: the 24 fps count, snapped up to the next `17n + 5`."""
    return align_num_frames(round(duration_s * MINIMAX_H3_FPS))


MINIMAX_H3_DURATIONS_S = tuple(range(4, 16))
MINIMAX_H3_DEFAULT_DURATION_S = 5
# Frame counts the shortest and longest served durations resolve to: 107 and 362.
MINIMAX_H3_MIN_NUM_FRAMES = get_num_frames(MINIMAX_H3_DURATIONS_S[0])
MINIMAX_H3_MAX_NUM_FRAMES = get_num_frames(MINIMAX_H3_DURATIONS_S[-1])

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

# Vision-tower patches at the reference caps: every image and every video block at the canvas area
# cap (4 patches per 32x32 merged token, 4,032 per block, which `match` never exceeds) and every
# video at the longest duration (16 blocks): 229,824.
MINIMAX_H3_MAX_REFERENCE_BLOCK_PATCHES = 4 * MINIMAX_H3_MAX_PIXELS // MINIMAX_H3_CANVAS_MULTIPLE**2
MINIMAX_H3_MAX_REFERENCE_PATCHES = MINIMAX_H3_MAX_REFERENCE_BLOCK_PATCHES * (
    MINIMAX_H3_MAX_REFERENCE_IMAGES
    + MINIMAX_H3_MAX_REFERENCE_VIDEOS
    * len(sample_reference_video_frames(np.zeros((MINIMAX_H3_MAX_NUM_FRAMES, 1, 1, 3), np.uint8))[1])
)

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


def validate_num_frames(num_frames: int) -> None:
    """`num_frames` must align into the frame counts the served durations produce, 4 s (107) to 15 s (362).

    Bounds are on the aligned count so a request of exactly 15 s, which aligns up to 15.083 s, is served.
    """
    aligned = align_num_frames(num_frames)
    if not MINIMAX_H3_MIN_NUM_FRAMES <= aligned <= MINIMAX_H3_MAX_NUM_FRAMES:
        raise ValueError(
            f"num_frames {num_frames} aligns to {aligned} ({aligned / MINIMAX_H3_FPS:.2f} s); served lengths are "
            f"{MINIMAX_H3_MIN_NUM_FRAMES} to {MINIMAX_H3_MAX_NUM_FRAMES} frames "
            f"({MINIMAX_H3_DURATIONS_S[0]} to {MINIMAX_H3_DURATIONS_S[-1]} s)"
        )


def validate_request(
    *,
    image: Image.Image | None,
    last_image: Image.Image | None,
    references: Sequence[MiniMaxH3Reference] | None,
    aspect_ratio: tuple[int, int],
    height: int | None,
    width: int | None,
    num_frames: int | None,
) -> None:
    """Reject a request outside the served envelope. A `num_frames` of None is left to the references."""
    validate_input(
        image=image,
        last_image=last_image,
        references=references,
        aspect_ratio=aspect_ratio,
        height=height,
        width=width,
    )
    if num_frames is not None:
        validate_num_frames(num_frames)


def validate_input(
    *,
    image: Image.Image | None,
    last_image: Image.Image | None,
    aspect_ratio: tuple[int, int],
    height: int | None,
    width: int | None,
    references: Sequence[MiniMaxH3Reference] | None = None,
) -> None:
    """Reject inputs outside the served envelope; references make it ref2va, any keyframe fl2va."""
    if (height is None) != (width is None):
        raise ValueError("pass both height and width, or neither")
    if references is not None:
        if image is not None or last_image is not None:
            raise ValueError("references (ref2va) and image/last_image (fl2va) are different tasks")
        validate_ref2va(aspect_ratio=aspect_ratio, height=height, width=width)
    elif image is None and last_image is None:
        validate_t2va(aspect_ratio=aspect_ratio, height=height, width=width)
    else:
        validate_fl2va(image=image, last_image=last_image, height=height, width=width)


def validate_t2va(*, aspect_ratio: tuple[int, int], height: int | None, width: int | None) -> None:
    """`aspect_ratio` must be a published ratio unless an explicit canvas replaces it."""
    _validate_canvas("T2VA", aspect_ratio, height, width)


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


def validate_ref2va(*, aspect_ratio: tuple[int, int], height: int | None, width: int | None) -> None:
    """`aspect_ratio` must be a published ratio unless an explicit canvas replaces it."""
    _validate_canvas("Ref2VA", aspect_ratio, height, width)


def _validate_canvas(task: str, aspect_ratio: tuple[int, int], height: int | None, width: int | None) -> None:
    if height is not None:
        _validate_explicit_canvas(height, width)
    elif aspect_ratio not in MINIMAX_H3_ASPECT_RATIOS:
        raise ValueError(
            f"aspect_ratio {aspect_ratio} is not served for {task}; supported: "
            + ", ".join(f"{w}:{h}" for w, h in MINIMAX_H3_ASPECT_RATIOS)
        )


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


def decodable_canvases() -> tuple[tuple[int, int], ...]:
    """Every `(height, width)` canvas `resolve_canvas_size` produces over the 1:4 to 4:1 range, sorted.

    A keyframe's canvas depends only on its aspect ratio, so holding the long side at the maximum and
    sweeping the short side covers every canvas, including those of the published ratios.
    """
    long_side = MINIMAX_H3_KEYFRAME_MAX_SIDE
    short_min = max(MINIMAX_H3_KEYFRAME_MIN_SIDE, -(-long_side // 4))
    canvases = set()
    for short in range(short_min, long_side + 1):
        canvases.add(resolve_canvas_size(long_side, short))
        canvases.add(resolve_canvas_size(short, long_side))
    return tuple(sorted(canvases))


# Vision-tower patches of the largest keyframe (4 per merged 32x32 patch). The tower's ring SDPA runs
# only for a single unpadded keyframe, so this bounds its K/V gather on the non-ref2va tasks.
MINIMAX_H3_MAX_DECODABLE_KEYFRAME_PATCHES = max(
    4 * (height // MINIMAX_H3_CANVAS_MULTIPLE) * (width // MINIMAX_H3_CANVAS_MULTIPLE)
    for height, width in decodable_canvases()
)


def served_keyframe_layouts(patch_alignment: int) -> tuple[tuple[int, tuple[int, int]], ...]:
    """One `(n_keyframes, canvas)` per vision-tower program set an fl2va request can reach.

    The tower pads the patch count to `patch_alignment`. It runs ring attention only for one keyframe
    that needs no pad; everything else is windowed, and windowed programs depend only on the padded
    size (the window count is kept out of the key).
    """
    canvases = decodable_canvases()
    multiple = MINIMAX_H3_CANVAS_MULTIPLE

    def patches(canvas: tuple[int, int]) -> int:
        return 4 * (canvas[0] // multiple) * (canvas[1] // multiple)

    by_key: dict[tuple[int, bool], tuple[int, tuple[int, int]]] = {}
    for n_keyframes in (1, 2):
        for canvas in sorted(canvases, key=lambda canvas: (patches(canvas), canvas)):
            total = n_keyframes * patches(canvas)
            padded = -(-total // patch_alignment) * patch_alignment
            by_key.setdefault((padded, n_keyframes == 1 and padded == total), (n_keyframes, canvas))
    return tuple(by_key[key] for key in sorted(by_key))


def served_envelope(task: str, *, patch_alignment: int | None = None) -> Iterator:
    """The vision-layout warm units a deployment must compile.

    t2va yields `(n_keyframes, canvas)` and needs the vision tower's `patch_alignment`
    (`sp_factor * TILE_SIZE`); ref2va yields `(canvas, image_size)`.
    """
    if task == "t2va":
        if patch_alignment is None:
            raise ValueError("served_envelope('t2va') needs patch_alignment")
        yield 0, None
        yield from served_keyframe_layouts(patch_alignment)
        return
    if task != "ref2va":
        raise NotImplementedError(f"served_envelope is not defined for task {task!r}")
    for canvas in served_reference_canvases():
        for size in served_reference_image_sizes(*canvas):
            yield canvas, size

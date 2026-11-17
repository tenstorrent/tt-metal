# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Matmul block sizes for the MiniMax-H3 all-gather-matmul shapes.

Keyed on `(K, N, per_core_M)`, since the best block depends on the M-tiles per core, which varies with
video duration. Passed to the linear layers as `default_block_size`, which takes precedence in `get_agmm_config`.

Two hard constraints the op asserts on:

`N_block` must be divisible by `subblock_w`, which `get_matmul_config` leaves at 2 -- so N_block must
be even. (For a fused-SwiGLU ff1 it must be even anyway: gate/up tile pairs interleave along N and a
block must never split a pair.)

`K_block` must divide `K_tiles_per_device` (`K / 32 / tp_factor`). The ring delivers the gathered K
in `K_block`-sized chunks and a partial final chunk is unsupported. With TP=4 that means K_block
divides 42 for K=5376 and 56 for K=7168 -- 8 (the generic default) divides neither, which is why
these shapes need an entry rather than falling back.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import ttnn

from ....utils.matmul import agmm_worker_grid
from .length_classes import LENGTH_CLASSES, length_class

# M parallelizes over 12 cores when transposed (M > N) and 10 otherwise (the op reserves the mux axis).
_TILE = 32
_M_CORES_TRANSPOSED = 12
_M_CORES_NON_TRANSPOSED = 10


def _per_core_m(m: int, n: int) -> int:
    """M-tiles each core walks, for an AGMM of M rows / N cols on the op's worker grid."""
    m_tiles = -(-m // _TILE)
    cores = _M_CORES_TRANSPOSED if m > n else _M_CORES_NON_TRANSPOSED
    return -(-m_tiles // cores)


# (K, N, per_core_M) -> (M_block, K_block, N_block). N is the *per-device* output width. Best block at
# subblock (2, 2), swept with models/tt_dit/utils/sweep_mm_block_sizes.py on 4x8 Blackhole Galaxy.
#   (5376, 5376)  attention to_qkv   K_tiles_per_device = 42
#   (7168, 1344)  attention to_out   K_tiles_per_device = 56
#   (5376, 7168)  feed-forward ff1   K_tiles_per_device = 42, fused SwiGLU so N_block must be even
AGMM_BLOCK_SIZES: dict[tuple[int, int, int], tuple[int, int, int]] = {
    (5376, 5376, 1): (2, 6, 16),
    (5376, 5376, 2): (2, 7, 16),
    (5376, 5376, 3): (4, 6, 16),
    (5376, 5376, 4): (6, 7, 16),
    (5376, 5376, 5): (6, 7, 16),
    (5376, 5376, 6): (6, 3, 16),
    (5376, 5376, 7): (4, 6, 16),
    (5376, 5376, 8): (4, 6, 16),
    (5376, 5376, 9): (6, 6, 16),
    (5376, 5376, 10): (6, 6, 16),
    (5376, 5376, 11): (4, 6, 16),
    (5376, 5376, 12): (4, 6, 16),
    (5376, 7168, 1): (2, 21, 6),
    (5376, 7168, 2): (2, 6, 12),
    (5376, 7168, 3): (4, 3, 14),
    (5376, 7168, 4): (4, 3, 16),
    (5376, 7168, 5): (6, 3, 16),
    (5376, 7168, 6): (6, 3, 16),
    (5376, 7168, 7): (4, 6, 14),
    (5376, 7168, 8): (4, 3, 16),
    (5376, 7168, 9): (6, 3, 14),
    (5376, 7168, 10): (6, 3, 16),
    (5376, 7168, 11): (6, 3, 16),
    (5376, 7168, 12): (4, 3, 16),
    (7168, 1344, 1): (2, 14, 6),
    (7168, 1344, 2): (2, 8, 6),
    (7168, 1344, 3): (6, 7, 4),
    (7168, 1344, 4): (8, 8, 6),
    (7168, 1344, 5): (6, 8, 6),
    (7168, 1344, 6): (10, 8, 6),
    (7168, 1344, 7): (8, 8, 6),
    (7168, 1344, 8): (8, 8, 6),
    (7168, 1344, 9): (10, 8, 6),
    (7168, 1344, 10): (10, 8, 6),
    (7168, 1344, 11): (6, 8, 6),
    (7168, 1344, 12): (6, 8, 8),
}


# Blocking of the dynamic-length path per (K, N) layer and length class; to_qkv and to_out take one
# blocking at every length, ff1 one per class.
AGMM_PINNED_BLOCKING: dict[tuple[int, int], dict[str, tuple[int, int, int]]] = {
    (5376, 5376): {"short": (4, 6, 16), "long": (4, 6, 16)},
    (5376, 7168): {"short": (6, 3, 16), "long": (4, 3, 16)},
    (7168, 1344): {"short": (6, 8, 6), "long": (6, 8, 6)},
}


class AgmmLayout(NamedTuple):
    """Worker-grid orientation, size and ring workers of an all-gather matmul pin."""

    transposed: bool
    grid: tuple[int, int]
    workers_per_link: int


# MiniMax-H3-Image overrides of the (video-tuned) pins above, per (K, N) and length class: (blocking, layout).
# Swept on a 4x8 BH Galaxy with test_matmul_sweep_image_minimax_h3.py over traffic-weighted image row counts per
# device (short: 544, 1056, 2112, 2560, 2976, 4608 -- 2K/4K text to image and edits; long: 7168, 9216, 14336 --
# many-reference edits) and adopted only where no row count got more than 3% slower than the video pin (one
# program serves every length in a class). Swept vs the video pin, weighted: ff1 short -20.5% (faster at every
# row count), ff1 long -9.3%, to_out long -12.3%. to_qkv and the short to_out keep the video pin: their best
# weighted candidates were slower at some served row count.
IMAGE_AGMM_PINS: dict[tuple[int, int], dict[str, tuple[tuple[int, int, int], AgmmLayout]]] = {
    (5376, 7168): {
        "short": ((4, 7, 10), AgmmLayout(transposed=False, grid=(12, 9), workers_per_link=6)),
        "long": ((6, 6, 14), AgmmLayout(transposed=True, grid=(12, 9), workers_per_link=7)),
    },
    (7168, 1344): {
        "long": ((2, 28, 4), AgmmLayout(transposed=False, grid=(12, 9), workers_per_link=5)),
    },
}
MATMUL_PROFILES = (None, "image")


class AgmmPin(NamedTuple):
    """Frozen all-gather-matmul program of one layer: grid, blocking and orientation fixed for every M."""

    config: ttnn.MinimalMatmulConfig
    num_workers_per_link: int
    transpose_core_grid: bool


# Worker-grid orientation per layer, measured on 4x8 Blackhole: to_qkv non-transposed is 21 % faster
# at 15 s than the transposed grid the op's `M > N` rule picks there, ff1 matches, to_out (N < M) transposed.
AGMM_PINNED_TRANSPOSED: dict[tuple[int, int], bool] = {
    (5376, 5376): False,
    (5376, 7168): False,
    (7168, 1344): True,
}


def agmm_pins(
    k: int, n: int, *, full_grid: ttnn.CoreCoord, num_links: int, profile: str | None = None
) -> dict[str, AgmmPin]:
    """The AGMM program of a `(K, N)` layer per length class: the video pin (one worker-grid orientation at every
    length and the class's `AGMM_PINNED_BLOCKING`), or with `profile="image"` the class's `IMAGE_AGMM_PINS` entry
    where there is one; classes with the same program share one pin."""
    if (k, n) not in AGMM_PINNED_BLOCKING:
        raise ValueError(f"dynamic_length has no pinned AGMM blocking for (K, N)={(k, n)}; TP=4 layouts only")
    if profile not in MATMUL_PROFILES:
        raise ValueError(f"unknown matmul profile {profile!r}; expected one of {MATMUL_PROFILES}")
    transposed = AGMM_PINNED_TRANSPOSED[(k, n)]
    grid = agmm_worker_grid(full_grid, transposed, num_links)
    # in0 senders group along the M-parallel axis (grid.x transposed, grid.y not) into `num_links` groups.
    in0_axis = grid.x if transposed else grid.y
    video_layout = AgmmLayout(transposed, (grid.x, grid.y), math.ceil(in0_axis / num_links))
    overrides = IMAGE_AGMM_PINS.get((k, n), {}) if profile == "image" else {}
    pins: dict[tuple, AgmmPin] = {}
    chosen: dict[str, tuple] = {}
    for cls in LENGTH_CLASSES:
        blocking, layout = overrides.get(cls.name, (AGMM_PINNED_BLOCKING[(k, n)][cls.name], video_layout))
        if layout.grid[0] > full_grid.x or layout.grid[1] > full_grid.y:
            raise ValueError(f"AGMM grid {layout.grid} exceeds the device grid {(full_grid.x, full_grid.y)}")
        key = (blocking, layout)
        if key not in pins:
            m_block, k_block, n_block = blocking
            config = ttnn.MinimalMatmulConfig(
                M_block_size=m_block,
                K_block_size=k_block,
                N_block_size=n_block,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(*layout.grid),
            )
            pins[key] = AgmmPin(config, layout.workers_per_link, layout.transposed)
        chosen[cls.name] = key
    return {name: pins[key] for name, key in chosen.items()}


def agmm_kwargs(pins: dict[str, AgmmPin] | None, k: int, n: int, rows: int) -> dict:
    """The linear's blocking kwargs for a `(K, N)` layer at `rows` per-device rows: the length class's pin,
    else (legacy path) the length-keyed swept table."""
    if pins is None:
        return {"default_block_size": agmm_block_size(k, n, rows)}
    pin = pins[length_class(rows).name]
    return {
        "matmul_config": pin.config,
        "num_workers_per_link": pin.num_workers_per_link,
        "transpose_core_grid": pin.transpose_core_grid,
    }


def agmm_block_size(k: int, n: int, m: int) -> tuple[int, int, int] | None:
    """Block sizes for an all-gather matmul of this `(K, N)` at sequence length `M`, or None to let
    the generic path decide. Uses the largest swept `per_core_M` that divides the runtime one.
    """
    per_core_m = _per_core_m(m, n)
    swept = [pcm for (kk, nn, pcm) in AGMM_BLOCK_SIZES if kk == k and nn == n]
    divisors = [pcm for pcm in swept if per_core_m % pcm == 0]
    if not divisors:
        return None
    return AGMM_BLOCK_SIZES[(k, n, max(divisors))]

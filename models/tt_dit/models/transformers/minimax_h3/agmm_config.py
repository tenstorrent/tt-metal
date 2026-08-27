# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Matmul block sizes for the MiniMax-H3 all-gather-matmul shapes.

Keyed on `(K, N, per_core_M)`. K and N are fixed by the architecture and the TP factor; the block
shape that wins, however, depends on how many M-tiles land on each core (`per_core_M`), which changes
with the requested video duration. So this table holds one entry per swept `per_core_M` operating
point, and `agmm_block_size` picks the right one for the runtime M.

`per_core_M` follows the op's own worker grid: the op reserves the mux axis, so M parallelizes over
12 cores when the output is narrow (transposed, M > N) and 10 cores otherwise. Passed to the linear
layers as `default_block_size`, which is precedence rule 1 in `get_agmm_config` -- a measured block
short-circuits both the swept `(M, K, N)` table and the v3 rule engine, so these apply without
registering anything globally. When `agmm_block_size` returns None the shape falls through to those.

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

# The op derives its worker grid from the device, reserving the mux axis: M parallelizes over 12
# cores when transposed (narrow output, M > N) and 10 otherwise. per_core_M -- the M-tiles each core
# walks -- follows.
_TILE = 32
_M_CORES_TRANSPOSED = 12
_M_CORES_NON_TRANSPOSED = 10


def _per_core_m(m: int, n: int) -> int:
    """M-tiles each core walks, for an AGMM of M rows / N cols on the op's worker grid."""
    m_tiles = -(-m // _TILE)  # ceil
    cores = _M_CORES_TRANSPOSED if m > n else _M_CORES_NON_TRANSPOSED
    return -(-m_tiles // cores)  # ceil


# (K, N, per_core_M) -> (M_block, K_block, N_block). N is the *per-device* output width; the key's
# per_core_M is the operating point the shape was swept at, and the value is the best block at
# subblock (2, 2) (what get_matmul_config applies to a default_block_size). Swept with
# models/tt_dit/utils/sweep_mm_block_sizes.py on 4x8 Blackhole Galaxy, one profiler session per
# shape, at the op's worker grid (non-transposed M over 10 cores / transposed over 12).
#   (5376, 5376)  attention to_qkv   K_tiles_per_device = 42
#   (7168, 1344)  attention to_out   K_tiles_per_device = 56
#   (5376, 7168)  feed-forward ff1   K_tiles_per_device = 42, fused SwiGLU so N_block must be even
# to_out per_core_M 1..3 are non-transposed (M < N), 4..12 transposed (M > N) -- agmm_block_size uses
# the same M>N test, so the runtime per_core_M lands on the matching entry across the crossover.
AGMM_BLOCK_SIZES: dict[tuple[int, int, int], tuple[int, int, int]] = {
    # (5376, 5376) attention to_qkv, non-transposed (M over 10)
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
    # (5376, 7168) feed-forward ff1, fused SwiGLU (N_block even), non-transposed (M over 10)
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
    # (7168, 1344) attention to_out (fused addcmul); per_core_M 1..3 non-transposed, 4..12 transposed
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


def agmm_block_size(k: int, n: int, m: int) -> tuple[int, int, int] | None:
    """Block sizes for an all-gather matmul of this `(K, N)` at sequence length `M`, or None to let
    the generic path decide.

    `M` sets `per_core_M` (how many M-tiles each core walks). Among the swept `per_core_M` entries
    for this `(K, N)`, pick the largest that divides the runtime `per_core_M` evenly -- so M tiles
    into whole blocks with no wasteful partial last block -- and return its block shape.
    """
    per_core_m = _per_core_m(m, n)
    swept = [pcm for (kk, nn, pcm) in AGMM_BLOCK_SIZES if kk == k and nn == n]
    divisors = [pcm for pcm in swept if per_core_m % pcm == 0]
    if not divisors:
        return None
    return AGMM_BLOCK_SIZES[(k, n, max(divisors))]

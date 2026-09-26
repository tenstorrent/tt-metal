# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Fused matmul + reduce-scatter (+ addcmul) blockings for the MiniMax-H3 feed-forward ff2.

`ff2` is row-parallel, so its reduce-scatter and the gated residual after it collapse into one
`minimal_matmul_strided_reduce_scatter_async`. That fusion is only a win with a *swept* blocking:
`get_fused_mmrs_config` falls back to `default_fused_mmrs_config`, which puts the matmul on an 8x7 =
56-core grid with `M_block=2` and subblock 1x1 -- half the machine at the least efficient subblock,
where the unfused path runs on 110 cores at subblock (2, 2). Measured, that fallback makes the fusion a
**45% regression** on the stage (1.75 -> 2.55 ms).

Unlike `agmm_config` these must be keyed on the full `(M, K, N)`, because that is what
`get_fused_mmrs_config` looks up -- there is no `default_block_size` hook to key more loosely. M is the
per-device packed sequence length, so it varies with the video duration; `has_mmrs_config` gates the
fused path off entirely for shapes it cannot serve, and `register_mmrs_config` registers the blocking
for whichever M the fused path is about to run at.

`compute_with_storage_grid_size` is the *matmul* grid, and the reduce-scatter workers occupy the rows
between it and the full device grid:

    rs_zone_capacity   = (device_grid.y - mm_grid.y) * device_grid.x
    num_workers_per_link = rs_zone_capacity // (2 * num_links) - 1

so the matmul grid and the reduce-scatter's bandwidth trade against each other directly, and that
trade is the dominant axis. Swept on 4x8 Blackhole Galaxy at 2 links with
`models/tt_dit/utils/sweep_mm_block_sizes.py` (`mmrs` use case), first across the three candidate
grids at M=4768:

    12x7   84 mm cores,  8 RS workers/link   M=6 K=4 N=16 sb(2,2)   1.373 ms
    12x8   96 mm cores,  5 RS workers/link   M=4 K=8 N=14 sb(2,2)   1.313 ms   <- best, -25.0%
    12x9  108 mm cores,  2 RS workers/link   M=6 K=2 N=8  sb(2,2)   1.487 ms

The optimum is interior: 12x9 starves the reduce-scatter and 12x7 starves the matmul. Ms without a
swept entry are NOT registered here: they fall through to the v2.3 rule engine inside
`get_fused_mmrs_config` (`utils/mmrs_rules.py`), which picks the 12x8 blocking per M. Measured
across the 41 swept H3 Ms, the rules average +1.9% over each M's swept optimum (worst +5.3%),
versus +6.6% (worst +26.7%, at small M) for the earlier policy of reusing the M=3424 blocking
at every duration.
"""

from __future__ import annotations

import ttnn

from ....utils.matmul import FusedMMRSConfig, register_fused_mmrs_configs, resolves_fused_mmrs_config

# ff2 per-device K = 14336 / tp 4 = 3584; N = 5376 is the full hidden size (the reduce-scatter
# fractures it back to 1344 per device on the way out).
_K = 3584
_N = 5376

# Per-M entries for the Ms that have been swept at their own shape; any other M is left
# unregistered so `get_fused_mmrs_config` resolves it with the v2.3 rules instead. Earlier
# revisions of this file keyed everything off M=4768 ("5s@768P"), then reused the nearest
# swept blocking for every duration; both traded up to 27% on Ms far from the swept one.
#
# M=3424 (the current run's per-device ff2 M) swept 2026-08-25 under the windowed L1 handoff,
# across all three candidate grids -- 12x8 still wins: 755.6 us vs 804.2 at 12x7 and 856.8 at
# 12x9. Mt_per_core = 107/12 -> 9, so M_block=8 leaves 2 blocks and the window rotates.
_SWEPT_BLOCKINGS = {
    3424: FusedMMRSConfig(ttnn.CoreCoord(12, 8), 8, 2, 8, 2, 2, None, 1),
}

_DEVICE_GRID = ttnn.CoreCoord(12, 10)

_TILE = 32
# The swept blocking's K divisibility holds for the fixed K: K_block 2 divides K/32 = 112. The
# rule engine guarantees the same for its own picks (K_block is chosen among divisors of K/32).


def has_mmrs_config(m: int, k: int, n: int, core_grid: ttnn.CoreCoord) -> bool:
    """Whether the fused MM+RS+addcmul path should be taken for this shape. Pure query, no side effects.

    Gate the fused path on this, and call `register_mmrs_config` before running the fused op. A shape
    this rejects would resolve neither a table entry nor a rule pick and land on
    `default_fused_mmrs_config`, whose 56-core matmul grid at subblock 1x1 makes the fused op far
    *slower* than not fusing -- measured as a 45% regression on this stage.

    Rather than a whitelist of the shipped durations, any tile-aligned M is accepted, so an
    arbitrary duration gets the fused path: a swept M resolves to its table entry, and every other
    M falls through to the v2.3 rules inside `get_fused_mmrs_config` (blind-validated within 5% of
    the swept optimum on 14/14 fresh H3 Ms). `M_block` does *not* have to divide the M tile count --
    a partial trailing block along M is fine, unlike along K, where the ring delivers the gathered
    input in fixed chunks; requiring divisibility here would silently disable the fused path for
    most durations.

    `core_grid` is the device's compute grid, and it is load-bearing rather than decorative: *both*
    ways of resolving a real blocking are architecture-specific -- `_SWEPT_BLOCKINGS` is only valid
    on `_DEVICE_GRID`, and the v2.3 rule engine is Blackhole-only. Gating on the shape alone accepted
    every tile-aligned M on Wormhole too, where neither source could hit, so all 50 ff2 blocks per
    denoise step silently ran the very fallback this gate exists to avoid. Ask
    `resolves_fused_mmrs_config` about the real grid instead of assuming the answer from (m, k, n).
    Wormhole now resolves through the global table's own 8x9 entries (`fused_mmrs_configs` in
    `utils/matmul.py`: the 15 s M = 13664 shape, swept 2026-09-23); every other Wormhole M stays on
    the unfused path until it is swept the same way.
    """
    if not (k == _K and n == _N and m % _TILE == 0):
        return False

    # The swept blocking is not in the global table until `register_mmrs_config` installs it, so
    # anticipate it here; otherwise the gate would answer on the pre-registration state.
    if m in _SWEPT_BLOCKINGS and core_grid == _DEVICE_GRID:
        return True

    return resolves_fused_mmrs_config(m, _K, _N, core_grid)


def register_mmrs_config(m: int, k: int, n: int, core_grid: ttnn.CoreCoord) -> None:
    """Register the swept blocking for this shape into the global fused-MMRS table, if one exists.

    Call at the site that is about to take the fused path (idempotent, cheap), for a shape
    `has_mmrs_config` accepts; registration is explicit so that merely *querying* a shape never
    mutates the global table. An M without a swept entry is deliberately NOT registered:
    `get_fused_mmrs_config` then resolves it with the v2.3 rule engine, which beats reusing a
    neighboring swept blocking (see the module docstring for the measured comparison).
    """
    if not has_mmrs_config(m, k, n, core_grid):
        msg = f"No fused MMRS blocking for (M, K, N) = ({m}, {k}, {n}); gate on has_mmrs_config first"
        raise ValueError(msg)
    # The swept blocking hardcodes a 12x8 matmul grid, so it is only meaningful on the grid it was
    # swept on -- registering it under any other key would hand that device a blocking off the end
    # of its own core grid.
    if m in _SWEPT_BLOCKINGS and core_grid == _DEVICE_GRID:
        register_fused_mmrs_configs({core_grid: {(m, _K, _N): _SWEPT_BLOCKINGS[m]}})

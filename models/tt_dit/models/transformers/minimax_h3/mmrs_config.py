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

The optimum is interior: 12x9 starves the reduce-scatter and 12x7 starves the matmul. The longer
durations reuse that blocking rather than being swept -- see `register_mmrs_config`.
"""

from __future__ import annotations

import ttnn

from ....utils.matmul import FusedMMRSConfig, register_fused_mmrs_configs

# ff2 per-device K = 14336 / tp 4 = 3584; N = 5376 is the full hidden size (the reduce-scatter
# fractures it back to 1344 per device on the way out).
_K = 3584
_N = 5376

# Only M=4768 (5s) was swept. Every other M reuses that blocking rather than being swept: K and N are
# fixed by the architecture, and M only sets how many blocks each core walks through, so the block
# *shape* has little reason to change with it. Sweeping per duration is also expensive -- warmup
# compiles one program per combo and compile time grows with M, putting M=9216 at ~75 min against ~9
# min at M=4768. What actually matters is that the fused op beats the unfused matmul + reduce-scatter +
# addcmul at each duration, and the block perf test measures that directly; see the perf log.
_SWEPT_BLOCKING = FusedMMRSConfig(ttnn.CoreCoord(12, 8), 4, 8, 14, 2, 2, None, 1)

_DEVICE_GRID = ttnn.CoreCoord(12, 10)

_TILE = 32
# The blocking's K and N divisibility hold for these fixed dims: K_block 8 divides K/32 = 112, and
# N_block 14 divides N/32 = 168.


def has_mmrs_config(m: int, k: int, n: int) -> bool:
    """Whether the fused MM+RS+addcmul path should be taken for this shape. Pure query, no side effects.

    Gate the fused path on this, and call `register_mmrs_config` before running the fused op. Where no
    config is registered, `get_fused_mmrs_config` falls back to `default_fused_mmrs_config`, whose
    56-core matmul grid at subblock 1x1 makes the fused op far *slower* than not fusing -- measured as
    a 45% regression on this stage.

    Rather than a whitelist of the shipped durations, any tile-aligned M is accepted and reuses the
    swept blocking, so an arbitrary duration gets the fused path. `M_block` does *not* have to divide
    the M tile count -- a partial trailing block along M is fine, unlike along K, where the ring
    delivers the gathered input in fixed chunks. M=4768 was swept successfully at `M_block=4` despite
    being 149 tiles; requiring divisibility here would silently disable the fused path at both 5s and 15s.
    """
    return k == _K and n == _N and m % _TILE == 0


def register_mmrs_config(m: int, k: int, n: int) -> None:
    """Register the swept blocking for this shape into the global fused-MMRS table.

    Call at the site that is about to take the fused path (idempotent, cheap), for a shape
    `has_mmrs_config` accepts; registration is explicit so that merely *querying* a shape never
    mutates the global table.
    """
    if not has_mmrs_config(m, k, n):
        msg = f"No fused MMRS blocking for (M, K, N) = ({m}, {k}, {n}); gate on has_mmrs_config first"
        raise ValueError(msg)
    register_fused_mmrs_configs({_DEVICE_GRID: {(m, _K, _N): _SWEPT_BLOCKING}})

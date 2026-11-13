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

import math
import os

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

# Blocking of the dynamic-length path: one config (the v2.3 rule pick over the served envelope) for
# every M, so a length never changes the fused op's program.
_PINNED_BLOCKING = FusedMMRSConfig(ttnn.CoreCoord(12, 8), 6, 4, 8, 2, 2, None, 1)

_TILE = 32
# The swept blocking's K divisibility holds for the fixed K: K_block 2 divides K/32 = 112. The
# rule engine guarantees the same for its own picks (K_block is chosen among divisors of K/32).


def pinned_mmrs_config(k: int, n: int, core_grid: ttnn.CoreCoord) -> FusedMMRSConfig | None:
    """The M-independent fused MM+RS blocking of ff2 on the grid it was fitted for, else None (unfused path)."""
    if (k, n) == (_K, _N) and core_grid == _DEVICE_GRID:
        return _PINNED_BLOCKING
    return None


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
    every tile-aligned M on Wormhole too, where neither source can hit, so all 50 ff2 blocks per
    denoise step silently ran the very fallback this gate exists to avoid. Ask
    `resolves_fused_mmrs_config` about the real grid instead of assuming the answer from (m, k, n).
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


_TO_OUT_K = 1792


def has_row_parallel_mmrs_config(m: int, k: int, n: int, core_grid: ttnn.CoreCoord) -> bool:
    """Whether to_out's fused MM+RS+addcmul should run for this shape (pure query). Same rule as `has_mmrs_config`:
    only when a swept entry or the rule engine resolves a blocking; the generic default is slower than not fusing."""
    if not (k == _TO_OUT_K and n == _N and m % _TILE == 0):
        return False
    return resolves_fused_mmrs_config(m, k, n, core_grid)


# Optional blocking overrides of the fused MM + RS on the legacy (length-keyed) path, default off. The reduce-scatter
# sends each output tile around the ring in the direction its position in the chunk picks (worker stride over the M
# block x N chunk), so a different M_block, N_block or matmul grid changes the order in which the TP partials of a
# tile are summed: these are a reassociation, not bit-identical, and stay behind a switch.
#   MINIMAX_H3_FF2_MMRS_BLOCKING      ff2    (K 3584): "1" = 12x8, M 10, K 4, N 7, subblock 2x1 (N 7 splits 14 columns
#                                     per core evenly; K_block as the rule pick)
#   MINIMAX_H3_TO_OUT_MMRS_BLOCKING   to_out (K 1792): "1" = 12x7 (one more reduce-scatter row), M 6, K 2, N 8, 2x2
#                                     (the rule pick's blocking)
# or an explicit "M,K,N,sub_h,sub_w,grid_y" (matmul grid 12 x grid_y) for in-model A/B. Unset, "" or "0": the rule pick.
# An override that would leave a core a single M block (no window rotation) is not applied at that M.
_MMRS_PRESETS = {
    "MINIMAX_H3_FF2_MMRS_BLOCKING": (10, 4, 7, 2, 1, 8),
    "MINIMAX_H3_TO_OUT_MMRS_BLOCKING": (6, 2, 8, 2, 2, 7),
}


def _mmrs_override_env(name: str) -> tuple[int, ...] | None:
    value = os.environ.get(name, "0").strip()
    if value in ("", "0"):
        return None
    if value == "1":
        return _MMRS_PRESETS[name]
    try:
        blocking = tuple(int(v) for v in value.split(","))
    except ValueError:
        blocking = ()
    if len(blocking) != 6 or min(blocking) < 1:
        raise ValueError(f"{name}={value!r}: expected 0, 1 or M,K,N,sub_h,sub_w,grid_y (positive integers)")
    m_block, _, n_block, sub_h, sub_w, _ = blocking
    if m_block % sub_h or n_block % sub_w:
        raise ValueError(f"{name}={value!r}: the subblock must divide M_block and N_block")
    return blocking


_MMRS_OVERRIDES = {
    _K: _mmrs_override_env("MINIMAX_H3_FF2_MMRS_BLOCKING"),
    _TO_OUT_K: _mmrs_override_env("MINIMAX_H3_TO_OUT_MMRS_BLOCKING"),
}
_MMRS_OVERRIDE_LOGGED: set[tuple[int, int]] = set()


def mmrs_blocking_override(m: int, k: int, n: int, core_grid: ttnn.CoreCoord) -> FusedMMRSConfig | None:
    """The switched-on blocking for ff2 / to_out's fused MM + RS at this shape, else None (the caller keeps the table /
    rule pick). Only for a shape the fused path already serves (has_mmrs_config / has_row_parallel_mmrs_config)."""
    blocking = _MMRS_OVERRIDES.get(k) if n == _N and core_grid == _DEVICE_GRID else None
    if blocking is None:
        return None
    m_block, k_block, n_block, sub_h, sub_w, grid_y = blocking
    if (k // _TILE) % k_block or not 1 <= grid_y < core_grid.y:
        raise ValueError(f"MMRS blocking override {blocking} does not fit K={k} on a {core_grid} device grid")
    rows_per_core = math.ceil(math.ceil(m / _TILE) / grid_y)
    if rows_per_core <= m_block:
        return None
    if (m, k) not in _MMRS_OVERRIDE_LOGGED:
        _MMRS_OVERRIDE_LOGGED.add((m, k))
        print(
            f"[mmrs] blocking override for (M, K, N)=({m}, {k}, {n}): grid {core_grid.x}x{grid_y}, "
            f"blocks {m_block, k_block, n_block}, subblock {sub_h}x{sub_w}",
            flush=True,
        )
    return FusedMMRSConfig(ttnn.CoreCoord(core_grid.x, grid_y), m_block, k_block, n_block, sub_h, sub_w, None, 1)

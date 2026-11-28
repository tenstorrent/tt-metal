# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Planned 2D-multicast program configs for the prefill linears (QKV / O / MLP gate_up / down; the 26B's interleaved
DecodeLinear weights and the dense models' DRAM-width-sharded DramShardedLinear weights alike).

A 4096-row prefill chunk on one Blackhole chip is compute-bound at HiFi2 (a 32x32x32 tile product ~32 FPU cycles):
the time is the busiest core's tile products, so the plan maximizes the cores that share the M x N output tiles
evenly. The candidates are every

* slab: M split into ``M / slab`` equal row slabs that the kernel iterates as a batch dim (a metadata reshape;
  the slab bounds the per-core output block the fp32 partials double in L1),
* grid: ``cols`` x ``rows`` of the device's compute grid with no idle row or column, N over the columns (an exact
  split for a DRAM-width-sharded weight, any split for an interleaved one), M over the rows,
* in0_block_w: the largest divisor of the K tiles up to 8 whose static CBs fit the L1 budget,

scored by an estimate of the busiest core's time, fitted on one Blackhole Galaxy chip (2026-11-18, the 12B / 26B
TP=4 shapes at 4096 rows): its tile products at the output subblock's rate, a fixed cost per K-block step
(multicast handshakes: larger in0_block_w, fewer steps), and the output write of the last slab, which nothing
overlaps (a second slab hides the first one's). The math is unchanged: the same HiFi2 + fp32 Dest compute config on
the same tiles; only the grouping of the fp32 K accumulation follows in0_block_w (outputs within one bf16 ulp of the
previous configs, PCC 1.0000000).

Two modes (``GEMMA4_PREFILL_MM_PLAN_KB``):

* ``keep`` (default): in0_block_w stays the one the caller's own config would use, so every output tile sums its K
  in the same blocks in the same order and the result is bitwise the caller's; only the slab, the grid and the
  output subblock move. Only callers that know their own K block take it (``DramShardedLinear``).
* ``free``: in0_block_w is planned too (the largest that fits): faster, but the fp32 K sum is grouped differently
  (outputs within one bf16 ulp, PCC 1.0000000 per op); every caller takes it.

``GEMMA4_PREFILL_MM_PLAN`` = 1 / 0 overrides the default (on for Blackhole, for the measured models' weights,
``PREFILL_MM_PLAN_DEFAULT_HIDDEN``). Calls below
``GEMMA4_PREFILL_MM_PLAN_MIN_ROWS`` rows (default 1024: the short-prompt buckets keep the configs their TTFT was
measured with) and rows that are not a tile multiple keep the caller's path.
"""

import functools
import math
import os

import ttnn
from models.common.utility_functions import is_blackhole

TILE = 32
PREFILL_MM_PLAN_ENV = "GEMMA4_PREFILL_MM_PLAN"
# Cost-model constants, Blackhole (ns): one HiFi2 tile product on a core by output subblock tile count, one K-block
# step, and the bytes / ns of an exposed output write.
_T_TILE_NS = {1: 45.0, 2: 34.6, 3: 33.5, 4: 32.5}
_T_STEP_NS = 2000.0
_T_SLAB_NS = 15000.0
_OUT_BYTES_PER_NS = 320.0
_MAX_KB = 17
_MAX_SLABS = 4
# hidden sizes where the plan is on by default: the 12B (3840). The 26B's (2816) dense projections are interleaved
# DecodeLinear weights, which plan only in the free mode (GEMMA4_PREFILL_MM_PLAN_KB=free), so 2816 is no default; the
# 31B (5376) and the rest keep their own configs unless GEMMA4_PREFILL_MM_PLAN=1
PREFILL_MM_PLAN_DEFAULT_HIDDEN = (3840,)
PREFILL_MM_PLAN_MIN_ROWS = int(os.environ.get("GEMMA4_PREFILL_MM_PLAN_MIN_ROWS", "1024"))


def prefill_mm_plan_keep_kb() -> bool:
    """``GEMMA4_PREFILL_MM_PLAN_KB``: ``keep`` (default; the caller's in0_block_w, bitwise) or ``free``."""
    mode = os.environ.get("GEMMA4_PREFILL_MM_PLAN_KB", "keep").strip().lower() or "keep"
    if mode not in ("keep", "free"):
        raise ValueError(f"GEMMA4_PREFILL_MM_PLAN_KB={mode!r}: expected keep or free")
    return mode == "keep"


def prefill_mm_plan_enabled(k=None, n=None) -> bool:
    """``GEMMA4_PREFILL_MM_PLAN`` when set, else on for Blackhole for a weight of one of the measured models' hidden
    sizes (``PREFILL_MM_PLAN_DEFAULT_HIDDEN``: ``k`` or ``n``)."""
    env = os.environ.get(PREFILL_MM_PLAN_ENV, "").strip().lower()
    if env:
        return env in ("1", "true", "yes", "on")
    return is_blackhole() and (k in PREFILL_MM_PLAN_DEFAULT_HIDDEN or n in PREFILL_MM_PLAN_DEFAULT_HIDDEN)


def cb_bytes(per_core_M, per_core_N, in0_block_w, weight_tile_bytes, fp32_acc):
    """Static CB bytes per core of the 2D prefill matmul (``dram_sharded.prefill_cb_bytes``)."""
    from models.demos.gemma4.tt.dram_sharded import prefill_cb_bytes

    return prefill_cb_bytes(per_core_M, per_core_N, in0_block_w, weight_tile_bytes, fp32_acc)


def best_subblock(per_core_M, per_core_N, dest_tiles=4):
    """(1, w): the widest one-row output subblock within the dest window (w divides per_core_N; the measured
    configs are all one row tall)."""
    return 1, max(w for w in range(1, min(dest_tiles, per_core_N) + 1) if per_core_N % w == 0)


@functools.lru_cache(maxsize=256)
def plan(m, k, n, grid, weight_tile_bytes, fp32_acc, budget, exact_n=True, max_kb=_MAX_KB, fixed_kb=None):
    """``(slab_rows, cols, rows, in0_block_w, sub_h, sub_w, per_core_M, per_core_N, est_ns)`` of the best candidate,
    or None when nothing fits (the caller keeps its own config). ``grid`` = (x, y) of the device compute grid.
    ``fixed_kb``: only that in0_block_w (the ``keep`` mode)."""
    gx, gy = int(grid[0]), int(grid[1])
    Mt, Kt, Nt = math.ceil(m / TILE), math.ceil(k / TILE), math.ceil(n / TILE)
    if Mt < 1 or Kt < 1 or Nt < 1:
        return None
    out_bytes = Mt * Nt * 2048
    best = None
    for nslab in range(1, _MAX_SLABS + 1):
        if Mt % nslab:
            continue
        slab_t = Mt // nslab
        for cols in range(1, gx + 1):
            pN = math.ceil(Nt / cols)
            if (exact_n and Nt % cols) or (cols - 1) * pN >= Nt:
                continue
            for rows in range(1, gy + 1):
                pM = math.ceil(slab_t / rows)
                if (rows - 1) * pM >= slab_t:
                    continue
                kbs = [int(fixed_kb)] if fixed_kb else range(min(Kt, max_kb), 0, -1)
                kb = next(
                    (d for d in kbs if Kt % d == 0 and cb_bytes(pM, pN, d, weight_tile_bytes, fp32_acc) <= budget),
                    None,
                )
                if kb is None:
                    continue
                h, w = best_subblock(pM, pN)
                compute = nslab * pM * pN * Kt * _T_TILE_NS[min(4, h * w)]
                steps = nslab * (Kt // kb) * _T_STEP_NS
                est = compute + steps + nslab * _T_SLAB_NS + out_bytes / nslab / _OUT_BYTES_PER_NS
                cand = (est, -cols * rows, nslab, slab_t * TILE, cols, rows, kb, h, w, pM, pN)
                if best is None or cand < best:
                    best = cand
    if best is None:
        return None
    est, _, _, slab, cols, rows, kb, h, w, pM, pN = best
    return slab, cols, rows, kb, h, w, pM, pN, est


def program_config(p, fused_activation=None):
    """The ``MatmulMultiCoreReuseMultiCastProgramConfig`` of a ``plan`` result."""
    slab, cols, rows, kb, h, w, pM, pN, _ = p
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=kb,
        out_subblock_h=h,
        out_subblock_w=w,
        per_core_M=pM,
        per_core_N=pN,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
    )


def planned_linear(
    x,
    weight,
    m,
    k,
    n,
    *,
    grid,
    weight_tile_bytes,
    fp32_acc,
    budget,
    exact_n,
    compute_kernel_config,
    memory_config,
    fixed_kb=None,
):
    """``x [1, 1, m, k] @ weight`` on the planned config (slabs as a batch dim, metadata reshapes), or None when no
    plan fits (the caller runs its own path). ``fixed_kb``: keep that in0_block_w (bitwise the caller's config)."""
    if m < PREFILL_MM_PLAN_MIN_ROWS or m % TILE or k % TILE:
        return None
    p = plan(m, k, n, tuple(grid), weight_tile_bytes, bool(fp32_acc), int(budget), bool(exact_n), fixed_kb=fixed_kb)
    if p is None:
        return None
    slab = p[0]
    pc = program_config(p)
    if slab == m:
        return ttnn.linear(
            x, weight, program_config=pc, compute_kernel_config=compute_kernel_config, memory_config=memory_config
        )
    x_r = ttnn.reshape(x, (1, m // slab, slab, k))
    out_r = ttnn.linear(
        x_r, weight, program_config=pc, compute_kernel_config=compute_kernel_config, memory_config=memory_config
    )
    return ttnn.reshape(out_r, (1, 1, m, int(out_r.shape[-1])))

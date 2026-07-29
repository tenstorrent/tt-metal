# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize perf bench — measurement only, NO correctness assertions.

Underscore-prefixed so pytest's default `test_*` collection ignores it and it
never enters the golden matrix. Run it explicitly:

    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py

Reports, per regime: the **device core count** the program actually launched on
(so lever A0 is machine-checkable rather than eyeballed), the median device
kernel duration over a trial loop, DRAM traffic, and achieved GB/s.

Ablation (`/perf-measure`) is driven by env flags read by the program
descriptor; output is garbage by design, which is why this file asserts no PCC:

    TILIZE_BENCH_ABLATE=1   # run all four variants per regime
      full        -> baseline
      no_compute  -> TILIZE_SKIP_COMPUTE=1 (CB dance kept, tilize LLK dropped)
      no_dm       -> TILIZE_SKIP_DM=1      (CB dance + barriers kept, NoC dropped)
      sync_only   -> both

Other knobs: TILIZE_BENCH_TRIALS (default 10), TILIZE_BENCH_REGIMES (comma-sep
regime names, default all).
"""

from __future__ import annotations

import os

# Enable the on-device profiler IN-PROCESS. All three are required by
# ttnn.get_latest_programs_perf_data() and must be set BEFORE the device opens.
# Module-scoped (not a dir conftest) so the op's correctness tests in this same
# directory are not perturbed. setdefault -> respects an outer tracy run.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd
from ttnn.operations.tilize.tilize_program_descriptor import (
    L1_CB_BUDGET_BYTES,
    L1_CB_BUDGET_PREFETCH_BYTES,
    a0_active_cores,
    build_plan,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

N_WARMUP = 3
N_TRIALS = int(os.environ.get("TILIZE_BENCH_TRIALS", "10"))  # launches per round
N_ROUNDS = int(os.environ.get("TILIZE_BENCH_ROUNDS", "5"))  # rounds -> median + CV
ABLATE = os.environ.get("TILIZE_BENCH_ABLATE", "0") == "1"
# One-sided DM ablation: decompose a serialized regime into its read and write legs.
# TILIZE_SKIP_DM=2 drops the read payload only, =3 the write payload only, so
# `full - no_read` prices the read leg and `full - no_write` the write leg.
SPLIT_DM = os.environ.get("TILIZE_BENCH_SPLIT_DM", "0") == "1"

_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR


def _crs(end_x, end_y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))})


def _shard(scheme, grid, shape):
    return ttnn.MemoryConfig(scheme, _L1, ttnn.ShardSpec(grid, shape, _ROW))


# name -> dict(shape, dtype, out_dtype, in_cfg, out_cfg, multicore, why)
# (a)-(f) are the mandatory regimes from op_design.md "Perf bench".
REGIMES = {
    # (a) grid-filling square — per-core DRAM efficiency once the grid is full.
    "a_square": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16),
    # (b) wide-short (nt_h=1, Wt=512) — THE gate: does the split fill the grid?
    #     A height-only split strands this on one core.
    "b_wide_short": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16),
    # (c) single-core reference baseline.
    "c_single_core": dict(shape=(1, 1, 512, 512), dtype=ttnn.bfloat16, multicore=False),
    # (d) tall-narrow guard — no-regression witness for the height regime.
    "d_tall_narrow": dict(shape=(1, 1, 2048, 32), dtype=ttnn.bfloat16),
    # (e) dtype sweep — page size changes the bound.
    "e_square_fp32": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.float32),
    "e_square_bf8b_out": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, out_dtype=ttnn.bfloat8_b),
    # (f) sharded, same spec (Path B, zero-copy) — small (~1 us) and large.
    "f_sharded_small": dict(
        shape=(1, 1, 512, 64),
        dtype=ttnn.bfloat16,
        in_cfg=_shard(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _crs(3, 0), (128, 64)),
        same_cfg=True,
    ),
    "f_sharded_large": dict(
        shape=(1, 1, 2048, 512),
        dtype=ttnn.bfloat16,
        in_cfg=_shard(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _crs(7, 7), (256, 64)),
        same_cfg=True,
    ),
    # (g) interleaved<->sharded crossover (generic path on both sides today; the
    #     design's R3b/R3c one-sided aliasing would take the sharded side's DRAM
    #     traffic to zero). NB: the traffic column counts bytes MOVED, not DRAM
    #     bytes — on a crossover one side lives in L1.
    "g_dram_to_sharded": dict(
        shape=(1, 1, 2048, 512),
        dtype=ttnn.bfloat16,
        out_cfg=_shard(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _crs(7, 7), (256, 64)),
    ),
    "g_sharded_to_dram": dict(
        shape=(1, 1, 2048, 512),
        dtype=ttnn.bfloat16,
        in_cfg=_shard(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _crs(7, 7), (256, 64)),
    ),
    # (e cont.) narrowing cast: fp32 in, bf16 out. The compute kernel picks
    # Fp32Mode::Lossless off the INPUT CB format, so this pays for the slow
    # tilize path even though the bf16 output cannot hold the extra precision.
    "e_square_fp32_to_bf16": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.float32, out_dtype=ttnn.bfloat16),
    # --- Mode-C counterfactuals: the same regime with ONE lever flipped off ----
    # C16 depth-2 CBs off -> reader/writer serialize instead of pipelining.
    # NB since Refinement 1 the *default* on this regime IS depth-1 (the gate),
    # so this row is the "explicitly forced depth-1" witness and
    # x_square_depth2 below is the counterfactual for the new default.
    "x_square_depth1": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, double_buffer=False),
    # C16 counterfactual for the gated default: force depth-2 back on where the
    # gate turned it off. `delta` here is what the gate costs/saves in ns; the
    # cbB/core column is what it saves in L1.
    "x_square_depth2": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, double_buffer=True),
    "x_tall_narrow_depth2": dict(shape=(1, 1, 2048, 32), dtype=ttnn.bfloat16, double_buffer=True),
    "x_single_core_depth1": dict(shape=(1, 1, 512, 512), dtype=ttnn.bfloat16, multicore=False, double_buffer=False),
    # The gate's blocks-per-core term is calibrated on these three: every regime
    # with >= 8 chunk-blocks per core, where depth-1's loss of read/write overlap
    # accumulates over the block loop. Paired IN-RUN with the auto rows above
    # (identical chunk width, so the only difference is the CB page count) --
    # cross-session comparison cannot resolve a ~2 % delta against the ~0.9 %
    # run-to-run scatter.
    "x_square_fp32_depth2": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.float32, double_buffer=True),
    "x_fp32_to_bf16_depth2": dict(
        shape=(1, 1, 2048, 2048), dtype=ttnn.float32, out_dtype=ttnn.bfloat16, double_buffer=True
    ),
    "x_sharded_to_dram_depth2": dict(
        shape=(1, 1, 2048, 512),
        dtype=ttnn.bfloat16,
        in_cfg=_shard(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _crs(7, 7), (256, 64)),
        double_buffer=True,
    ),
    "x_square_bf8b_depth2": dict(
        shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, out_dtype=ttnn.bfloat8_b, double_buffer=True
    ),
    # A0 counterfactual: force the ~16-core dram_saturation bandwidth knee as a
    # core cap on the regime it was proposed for. Refinement 1 measured this
    # 2.4x SLOWER than the full grid (the op is read-transaction-rate bound, so
    # the bandwidth knee never binds) -- kept as a bench row so the verdict is
    # re-measurable rather than a claim in a changelog.
    "x_tall_narrow_16c": dict(shape=(1, 1, 2048, 32), dtype=ttnn.bfloat16, core_cap=16),
    # A0 2D split off -> the wide-short shape (nt_h=1) collapses onto one core,
    # which is exactly what a height-only split_work_to_cores(nt_h) would do.
    "x_wide_short_1core": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, multicore=False),
    # --- Refinement 1c: per-lever counterfactuals on the sub-one-packet read
    # path (B13 stateful bank-major reads, C7 split reader). Every row below is
    # measured IN THE SAME RUN as the regime it is the counterfactual for, because
    # a few-percent delta is not resolvable against the cross-session scatter.
    # `levers=dict(b13=2)` / `c7=2` FORCE the lever past its own payoff gate --
    # that is what makes the gate itself re-measurable instead of a claim.
    #
    # 64 B reads, 1 block/core: both levers pay (this is the target regime).
    "x_tall_narrow_no_levers": dict(shape=(1, 1, 2048, 32), dtype=ttnn.bfloat16, levers=dict(b13=0, c7=0, b8=0)),
    "x_tall_narrow_b13_only": dict(shape=(1, 1, 2048, 32), dtype=ttnn.bfloat16, levers=dict(b13=1, c7=0)),
    "x_tall_narrow_c7_only": dict(shape=(1, 1, 2048, 32), dtype=ttnn.bfloat16, levers=dict(b13=0, c7=1)),
    # 64 B reads, 4 blocks/core: B13 still pays, C7 turns over (it spends the
    # read/write overlap across the block boundary). Default here is B13 only.
    "n_tall_narrow_4blk": dict(shape=(1, 1, 8192, 32), dtype=ttnn.bfloat16),
    "x_tall_narrow_4blk_no_levers": dict(shape=(1, 1, 8192, 32), dtype=ttnn.bfloat16, levers=dict(b13=0, c7=0, b8=0)),
    "x_tall_narrow_4blk_c7_forced": dict(shape=(1, 1, 8192, 32), dtype=ttnn.bfloat16, levers=dict(b13=0, c7=2)),
    # Read-size sweep at a fixed 64 cores x 1 block/core (nt_h == 1, Wt = 128/256
    # makes the planner pick chunk_wt 2/4 => 128 B / 256 B reads). 128 B is the
    # largest read where B13 still pays; C7 is already negative there.
    "m_wide_short_4k": dict(shape=(1, 1, 32, 4096), dtype=ttnn.bfloat16),
    "x_wide_short_4k_no_levers": dict(shape=(1, 1, 32, 4096), dtype=ttnn.bfloat16, levers=dict(b13=0, c7=0)),
    "x_wide_short_4k_c7_forced": dict(shape=(1, 1, 32, 4096), dtype=ttnn.bfloat16, levers=dict(b13=0, c7=2)),
    "m_wide_short_8k": dict(shape=(1, 1, 32, 8192), dtype=ttnn.bfloat16),
    "x_wide_short_8k_b13_forced": dict(shape=(1, 1, 32, 8192), dtype=ttnn.bfloat16, levers=dict(b13=2, c7=0)),
    "x_wide_short_8k_c7_forced": dict(shape=(1, 1, 32, 8192), dtype=ttnn.bfloat16, levers=dict(b13=0, c7=2)),
    # 512 B (b_wide_short) is the worst case for both levers, and 1024 B
    # (g_dram_to_sharded, a_square) is where B13 costs ~5 % on the crossover and
    # disappears into the DRAM-bandwidth floor on the square. All three keep BOTH
    # levers off by default; these rows are what that decision is based on.
    "x_wide_short_b13_forced": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(b13=2, c7=0)),
    "x_wide_short_c7_forced": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(b13=0, c7=2)),
    "x_square_b13_forced": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, levers=dict(b13=2, c7=0)),
    "x_g_to_sharded_b13_forced": dict(
        shape=(1, 1, 2048, 512),
        dtype=ttnn.bfloat16,
        out_cfg=_shard(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _crs(7, 7), (256, 64)),
        levers=dict(b13=2, c7=0),
    ),
    "x_g_to_sharded_c7_forced": dict(
        shape=(1, 1, 2048, 512),
        dtype=ttnn.bfloat16,
        out_cfg=_shard(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _crs(7, 7), (256, 64)),
        levers=dict(b13=0, c7=2),
    ),
    # --- Refinement 2: per-lever counterfactuals for B8 (trid double-issue),
    # B10 (per-core static unicast VC) and A3 (bank-adjacent work->core order).
    # `levers=dict(b8=N)`: 0 = off, 1 = gated, 2 = FORCE past the payoff gate,
    # 3 = the extra CB window WITHOUT the trid pipeline (isolation row, so the
    # ledger can separate "deeper CB" from "reads in flight across the barrier").
    #
    # B8's own regime is low-core-count + multi-block. c_single_core (1 core,
    # 16 blk) and x_wide_short_1core (1 core, 32 blk) are where the gate turns it
    # ON, so those two need the 3-way (off / window-only / full) comparison.
    "x_single_core_b8_off": dict(shape=(1, 1, 512, 512), dtype=ttnn.bfloat16, multicore=False, levers=dict(b8=0)),
    "x_single_core_b8_window_only": dict(
        shape=(1, 1, 512, 512), dtype=ttnn.bfloat16, multicore=False, levers=dict(b8=3)
    ),
    "x_wide_short_1core_b8_off": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, multicore=False, levers=dict(b8=0)),
    "x_wide_short_1core_b8_window_only": dict(
        shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, multicore=False, levers=dict(b8=3)
    ),
    # ... and forcing it past the gate on the full grid is the counterfactual for
    # the gate's core-count clause (a_square 4 blk/core, n_tall_narrow_4blk 4 blk).
    "x_square_b8_forced": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, levers=dict(b8=2)),
    # B8 read-transaction-size sweep at a FIXED 64 cores x 2 blocks/core (nt_h = 128
    # with Wt = 2/4/8/16 makes the planner pick chunk 2/4/8/16 => 128/256/512/1024 B
    # reads while `n_chunks` stays 1). This is what sets the gate's size clause: the
    # first sweep found B8 worth -19 % at 64 B and inert at 1024 B on the same core
    # count, so the threshold is a read SIZE, not only a core count.
    # The gated DEFAULT on the 128 B / 2-blk regime -- the shipped number future
    # phases must not regress (the `p_*` row below is its counterfactual).
    "m_2blk_128B": dict(shape=(1, 1, 4096, 64), dtype=ttnn.bfloat16),
    "p_2blk_128B": dict(shape=(1, 1, 4096, 64), dtype=ttnn.bfloat16, levers=dict(b8=0, b13=0)),
    "x_2blk_128B_b8": dict(shape=(1, 1, 4096, 64), dtype=ttnn.bfloat16, levers=dict(b8=2, b13=0)),
    "p_2blk_256B": dict(shape=(1, 1, 4096, 128), dtype=ttnn.bfloat16, levers=dict(b8=0, b13=0)),
    "x_2blk_256B_b8": dict(shape=(1, 1, 4096, 128), dtype=ttnn.bfloat16, levers=dict(b8=2, b13=0)),
    "p_2blk_512B": dict(shape=(1, 1, 4096, 256), dtype=ttnn.bfloat16, levers=dict(b8=0, b13=0)),
    "x_2blk_512B_b8": dict(shape=(1, 1, 4096, 256), dtype=ttnn.bfloat16, levers=dict(b8=2, b13=0)),
    "p_2blk_1024B": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, levers=dict(b8=0, b13=0)),
    "x_2blk_1024B_b8": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, levers=dict(b8=2, b13=0)),
    # 64 B x 4 blocks at 64 cores -- the row where the first sweep found B8 beats
    # even B13 (which the planner ships there today).
    "x_tall_narrow_4blk_b13_only": dict(shape=(1, 1, 8192, 32), dtype=ttnn.bfloat16, levers=dict(b8=0, b13=1)),
    # B8 CORE-COUNT sweep at a fixed 1024 B read (the size where B8 is inert on the
    # full grid but pays 10-12 % on ONE core). `core_cap` forces the count through
    # the planner's sweep hook, so chunk stays 16 and only `ncores` moves. This is
    # what sets the gate's core clause instead of borrowing BANDWIDTH_KNEE_CORES.
    "p_1024B_1c": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, core_cap=1, levers=dict(b8=0)),
    "x_1024B_1c_b8": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, core_cap=1, levers=dict(b8=2)),
    "p_1024B_2c": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, core_cap=2, levers=dict(b8=0)),
    "x_1024B_2c_b8": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, core_cap=2, levers=dict(b8=2)),
    "p_1024B_4c": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, core_cap=4, levers=dict(b8=0)),
    "x_1024B_4c_b8": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, core_cap=4, levers=dict(b8=2)),
    "p_1024B_8c": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, core_cap=8, levers=dict(b8=0)),
    "x_1024B_8c_b8": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, core_cap=8, levers=dict(b8=2)),
    "p_1024B_16c": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, core_cap=16, levers=dict(b8=0)),
    "x_1024B_16c_b8": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, core_cap=16, levers=dict(b8=2)),
    "p_1024B_32c": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, core_cap=32, levers=dict(b8=0)),
    "x_1024B_32c_b8": dict(shape=(1, 1, 4096, 512), dtype=ttnn.bfloat16, core_cap=32, levers=dict(b8=2)),
    "x_tall_narrow_4blk_b8_forced": dict(shape=(1, 1, 8192, 32), dtype=ttnn.bfloat16, levers=dict(b8=2, b13=0)),
    "x_fp32_b8_forced": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.float32, levers=dict(b8=2)),
    # B10: VC diversity can only break a queue that several cores share, so the
    # interesting regimes are the full-grid ones. c_single_core is the B0 control
    # (one core cannot contend with itself).
    "x_wide_short_b10_forced": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(b10=2)),
    # ... and the two halves separately (b10=3 reads only, b10=4 writes only): the
    # read VC is a sticky NOC_CTRL program and the write VC is a per-call field, so
    # a single "B10" number cannot say which mechanism moved the clock.
    "x_wide_short_b10_read_only": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(b10=3)),
    "x_wide_short_b10_write_only": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(b10=4)),
    "x_square_b10_read_only": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, levers=dict(b10=3)),
    "x_square_b10_write_only": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, levers=dict(b10=4)),
    "x_square_b10_forced": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, levers=dict(b10=2)),
    "x_tall_narrow_b10_forced": dict(shape=(1, 1, 2048, 32), dtype=ttnn.bfloat16, levers=dict(b10=2)),
    "x_single_core_b10_forced": dict(shape=(1, 1, 512, 512), dtype=ttnn.bfloat16, multicore=False, levers=dict(b10=2)),
    "x_g_to_sharded_b10_forced": dict(
        shape=(1, 1, 2048, 512),
        dtype=ttnn.bfloat16,
        out_cfg=_shard(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _crs(7, 7), (256, 64)),
        levers=dict(b10=2),
    ),
    # A3: host-only work->core permutation. Same regimes as B10 (it is the other
    # congestion lever), plus the wide-short one it was proposed for.
    "x_wide_short_a3_forced": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(a3=2)),
    "x_square_a3_forced": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, levers=dict(a3=2)),
    "x_tall_narrow_a3_forced": dict(shape=(1, 1, 2048, 32), dtype=ttnn.bfloat16, levers=dict(a3=2)),
    # B10 + A3 together: both attack route congestion, so the bundle is the row
    # that answers "did we only fail to see it because each alone is too small?"
    "x_wide_short_b10_a3_forced": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(b10=2, a3=2)),
    # --- Refinement 2b: the wide-short 64-way partial-page fan-in --------------
    # `levers=dict(r2b=N)`: 0 = off, 1 = gated, 2 = FORCE the full whole-page-read +
    # L1-redistribution algorithm past its payoff gate, 3 = the MEASUREMENT PROBE
    # (phase 1 only -- one whole-piece read per core, no exchange). The probe prices
    # the read-side ceiling on its own: it moves the same bytes with the same cores
    # in 1 transaction instead of 32, so `probe/off` is the most this algorithm can
    # ever buy and `forced/probe` is what the extra L1 hop + barrier costs.
    "p_wide_short_r2b_off": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(r2b=0, stg=0)),
    "x_wide_short_r2b_probe": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(r2b=3, stg=0)),
    "x_wide_short_r2b_forced": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(r2b=2, stg=0)),
    # ... and on the two narrower members of the same family, where the fan-in slice
    # is 256 B / 128 B so the staged read is 8192 B / 4096 B instead of 16384 B.
    "p_wide_short_8k_r2b_off": dict(shape=(1, 1, 32, 8192), dtype=ttnn.bfloat16, levers=dict(r2b=0, b13=0, stg=0)),
    "x_wide_short_8k_r2b_probe": dict(shape=(1, 1, 32, 8192), dtype=ttnn.bfloat16, levers=dict(r2b=3, b13=0, stg=0)),
    "x_wide_short_8k_r2b_forced": dict(shape=(1, 1, 32, 8192), dtype=ttnn.bfloat16, levers=dict(r2b=2, b13=0, stg=0)),
    "p_wide_short_4k_r2b_off": dict(shape=(1, 1, 32, 4096), dtype=ttnn.bfloat16, levers=dict(r2b=0, b13=0, stg=0)),
    "x_wide_short_4k_r2b_probe": dict(shape=(1, 1, 32, 4096), dtype=ttnn.bfloat16, levers=dict(r2b=3, b13=0, stg=0)),
    "x_wide_short_4k_r2b_forced": dict(shape=(1, 1, 32, 4096), dtype=ttnn.bfloat16, levers=dict(r2b=2, b13=0, stg=0)),
    # --- Refinement 2b, second lever: read/write OVERLAP on the wide-short regime.
    # The one-sided DM decomposition of b_wide_short (TILIZE_BENCH_SPLIT_DM=1) prices
    # the read leg at 5 966 ns and the WRITE leg at 7 751 ns, while the whole op takes
    # 13 461 -- i.e. the two legs overlap by only 2 482 of a possible ~5 966 ns,
    # because `nt_h == 1` gives every core exactly ONE chunk-block and a single block
    # has no successor to overlap with. `chunk_cap` forces a narrower chunk => MORE
    # blocks per core at the SAME 64 cores, which is the only way to create a block
    # boundary on this shape. The cost is a smaller read transaction
    # (512 -> 256/128/64 B); the sweep is what decides whether the overlap pays for it.
    # All rows keep B13/B8 off so the blocking effect is isolated; the `*_gated` rows
    # add them back at the sizes where their own gates fire.
    "p_wide_short_chunk8": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(r2b=0, b13=0, b8=0)),
    "x_wide_short_chunk4": dict(
        shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, chunk_cap=4, levers=dict(r2b=0, b13=0, b8=0)
    ),
    "x_wide_short_chunk2": dict(
        shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, chunk_cap=2, levers=dict(r2b=0, b13=0, b8=0)
    ),
    "x_wide_short_chunk1": dict(
        shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, chunk_cap=1, levers=dict(r2b=0, b13=0, b8=0)
    ),
    "x_wide_short_chunk4_d2": dict(
        shape=(1, 1, 32, 16384),
        dtype=ttnn.bfloat16,
        chunk_cap=4,
        double_buffer=True,
        levers=dict(r2b=0, b13=0, b8=0),
    ),
    "x_wide_short_chunk2_d2": dict(
        shape=(1, 1, 32, 16384),
        dtype=ttnn.bfloat16,
        chunk_cap=2,
        double_buffer=True,
        levers=dict(r2b=0, b13=0, b8=0),
    ),
    "x_wide_short_chunk1_d2": dict(
        shape=(1, 1, 32, 16384),
        dtype=ttnn.bfloat16,
        chunk_cap=1,
        double_buffer=True,
        levers=dict(r2b=0, b13=0, b8=0),
    ),
    "x_wide_short_chunk2_gated": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, chunk_cap=2, levers=dict(r2b=0)),
    "x_wide_short_chunk1_gated": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, chunk_cap=1, levers=dict(r2b=0)),
    # --- Refinement 2b, per-core transaction-order rotation (`stg`) -------------
    # 0 = off, 1 = gated, 2 = forced. Read and write rotations are decided by the
    # SAME flag in the planner, so the isolation rows use the one-sided DM ablation
    # (TILIZE_BENCH_SPLIT_DM=1) to attribute the delta to a leg.
    "p_wide_short_stg_off": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(r2b=0, stg=0)),
    "x_wide_short_stg": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(r2b=0, stg=2)),
    "x_wide_short_stg_read_only": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(r2b=0, stg=3)),
    "x_wide_short_stg_write_only": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, levers=dict(r2b=0, stg=4)),
    "x_wide_short_8k_stg_read_only": dict(
        shape=(1, 1, 32, 8192), dtype=ttnn.bfloat16, levers=dict(r2b=0, b13=0, stg=3)
    ),
    "x_wide_short_8k_stg_write_only": dict(
        shape=(1, 1, 32, 8192), dtype=ttnn.bfloat16, levers=dict(r2b=0, b13=0, stg=4)
    ),
    "p_wide_short_4k_stg_off": dict(shape=(1, 1, 32, 4096), dtype=ttnn.bfloat16, levers=dict(r2b=0, b13=0, stg=0)),
    "x_wide_short_4k_stg": dict(shape=(1, 1, 32, 4096), dtype=ttnn.bfloat16, levers=dict(r2b=0, b13=0, stg=2)),
    # Gate sweep: the read-side clustering exists whenever the tile-row is split by
    # COLUMN (`n_w > 1`), so several cores read the SAME source pages in the same
    # order. These two rows extend the sweep past `nt_h == 1`: chunk 16 at nt_h = 1,
    # and nt_h = 2 (n_w = 32, so 32 cores share each row group).
    "p_wide_short_32k_stg_off": dict(shape=(1, 1, 32, 32768), dtype=ttnn.bfloat16, levers=dict(r2b=0, stg=0)),
    "n_wide_short_32k": dict(shape=(1, 1, 32, 32768), dtype=ttnn.bfloat16, levers=dict(r2b=0, stg=2)),
    "p_wide_short_2row_stg_off": dict(shape=(1, 1, 64, 16384), dtype=ttnn.bfloat16, levers=dict(r2b=0, stg=0)),
    "n_wide_short_2row": dict(shape=(1, 1, 64, 16384), dtype=ttnn.bfloat16, levers=dict(r2b=0, stg=2)),
    # Can the stagger + re-blocking be combined? Re-blocking alone was 1.019 / 1.153
    # (chunk 4 / 2) because the per-block sync floor grows ~400-500 ns per block; the
    # question is whether the read/write overlap a second block buys is worth more
    # once the banks are de-clustered.
    "x_wide_short_chunk4_stg": dict(
        shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, chunk_cap=4, levers=dict(r2b=0, b13=0, b8=0, stg=2)
    ),
    "x_wide_short_chunk2_stg": dict(
        shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, chunk_cap=2, levers=dict(r2b=0, b13=0, b8=0, stg=2)
    ),
    # Rotation-modulus sweep: TILE_HW (32, the row-loop period) vs NUM_DRAM_BANKS (12,
    # which makes the per-core starting bank perfectly uniform).
    "x_wide_short_stg_mod12": dict(
        shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, stagger_mod=12, levers=dict(r2b=0, stg=2)
    ),
    "x_wide_short_8k_stg_mod12": dict(
        shape=(1, 1, 32, 8192), dtype=ttnn.bfloat16, stagger_mod=12, levers=dict(r2b=0, b13=0, stg=2)
    ),
    "p_square_stg_off": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, levers=dict(stg=0)),
    "x_square_stg_read_only": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, levers=dict(stg=3)),
    "x_square_stg_write_only": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, levers=dict(stg=4)),
    "p_g_to_sharded_stg_off": dict(
        shape=(1, 1, 2048, 512),
        dtype=ttnn.bfloat16,
        out_cfg=_shard(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _crs(7, 7), (256, 64)),
        levers=dict(stg=0),
    ),
    "x_g_to_sharded_stg": dict(
        shape=(1, 1, 2048, 512),
        dtype=ttnn.bfloat16,
        out_cfg=_shard(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _crs(7, 7), (256, 64)),
        levers=dict(stg=2),
    ),
    "p_square_fp32_stg_off": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.float32, levers=dict(stg=0)),
    "x_square_fp32_stg": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.float32, levers=dict(stg=2)),
    "x_square_stg": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, levers=dict(stg=2)),
    "p_tall_narrow_stg_off": dict(shape=(1, 1, 2048, 32), dtype=ttnn.bfloat16, levers=dict(b13=0, c7=0, stg=0)),
    "x_tall_narrow_stg": dict(shape=(1, 1, 2048, 32), dtype=ttnn.bfloat16, levers=dict(b13=0, c7=0, stg=2)),
    "p_wide_short_8k_stg_off": dict(shape=(1, 1, 32, 8192), dtype=ttnn.bfloat16, levers=dict(r2b=0, b13=0, stg=0)),
    "x_wide_short_8k_stg": dict(shape=(1, 1, 32, 8192), dtype=ttnn.bfloat16, levers=dict(r2b=0, b13=0, stg=2)),
    # C16 on the smallest sharded regime (lever B0: per-core-overhead levers must
    # be counterfactualed on the SMALLEST shape they run in).
    "x_sharded_small_depth1": dict(
        shape=(1, 1, 512, 64),
        dtype=ttnn.bfloat16,
        in_cfg=_shard(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _crs(3, 0), (128, 64)),
        same_cfg=True,
        double_buffer=False,
    ),
}

_SELECTED = os.environ.get("TILIZE_BENCH_REGIMES", "")
REGIME_NAMES = [n.strip() for n in _SELECTED.split(",") if n.strip()] or list(REGIMES)


def _read_kernel_ns(device):
    """Summed on-device kernel duration for programs dispatched since the last read.

    ReadDeviceProfiler flushes the queue and *consumes* the window, so a
    flush-read then a work-read brackets exactly the launches in between.
    """
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure_median_ns(device, run_fn):
    """Median ns/launch over N_ROUNDS rounds of N_TRIALS launches each.

    Reads are BATCHED per round on purpose: ReadDeviceProfiler after a single
    launch reliably returns an empty window on this build, so each round runs
    N_TRIALS launches and divides. Warm-up window is flushed and discarded.
    Rounds give the std-dev / CV that the /perf-measure noise threshold needs.
    """
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # drop the warm-up window

    samples = []
    for _ in range(N_ROUNDS):
        for _ in range(N_TRIALS):
            run_fn()
        value = _read_kernel_ns(device)
        if value is not None:
            samples.append(value / N_TRIALS)
    if not samples:
        return None, None
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return statistics.median(samples), std


def _build(device, spec):
    shape = spec["shape"]
    dtype = spec["dtype"]
    in_cfg = spec.get("in_cfg", ttnn.DRAM_MEMORY_CONFIG)
    out_cfg = in_cfg if spec.get("same_cfg") else spec.get("out_cfg", ttnn.DRAM_MEMORY_CONFIG)

    torch.manual_seed(0)
    if dtype == ttnn.float32:
        torch_input = torch.randn(shape, dtype=torch.float32)
    else:
        torch_input = torch.randn(shape).bfloat16()

    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
    )
    return tt_input, out_cfg


def _plan_for(device, tt_input, spec, out_cfg):
    """Rebuild the plan host-side to report ncores / chunk_wt / depth / CB bytes.

    ``double_buffer`` defaults to ``None`` == the op's *gated* default, so the
    bench measures what a caller actually gets (Refinement 1, lever C16).
    """
    out_dtype = spec.get("out_dtype") or tt_input.dtype
    probe_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(tt_input.shape)), out_dtype, ttnn.TILE_LAYOUT, device, out_cfg
    )
    return build_plan(
        tt_input,
        probe_out,
        device,
        use_multicore=spec.get("multicore", True),
        use_double_buffer=spec.get("double_buffer"),
    )


if ABLATE:
    _VARIANTS = [("full", "0", "0"), ("no_compute", "0", "1"), ("no_dm", "1", "0"), ("sync_only", "1", "1")]
elif SPLIT_DM:
    _VARIANTS = [("full", "0", "0"), ("no_read", "2", "0"), ("no_write", "3", "0"), ("no_dm", "1", "0")]
else:
    _VARIANTS = [("full", "0", "0")]


def _assert_structural_gates(name, spec, plan, grid_cores):
    """Machine-check the *structural* perf gates (no correctness involved).

    A0 (``master.md`` Part 2 §A) states the criterion as
    ``active == min(grid, total_tiles, bandwidth_knee)`` — the **gated** form, with
    the knee term included (Refinement 1). The gate lives in the op
    (``a0_active_cores``); this asserts the criterion independently, so a
    height-only-split regression on the wide-short regime (``nt_h == 1``, which
    would strand it on ONE core while the duration column still looks healthy)
    still fails the bench. On the zero-copy alias path the criterion is the
    shard's own cores; ``use_multicore=False`` is exactly 1.

    Refinement 1 measured tilize's own knee at the full grid (the op is
    read-transaction-rate bound, not DRAM-bandwidth bound — capping at the
    16-core ``dram_saturation`` knee is 2.4x SLOWER, see ``x_tall_narrow_16c``),
    so ``A0_KNEE_CORES`` is identity here and the assert below reduces to
    ``min(grid, total_tiles)`` on any current compute grid. Lowering that
    constant without re-running ``probes/probe_009.py`` will trip this assert.

    Bounded CB: per-core CB L1 must stay inside the planner's budget, i.e. be a
    constant in ``W`` — the claim ``PROPERTIES["bounded_cb"]`` makes.

    Gated depth (C16): the depth the planner picked must match the declared gate
    whenever the regime does not force ``use_double_buffer``.
    """
    if plan["path"] == "alias":
        expected = plan["ncores"]  # the shard's own cores, by construction
    elif not spec.get("multicore", True):
        expected = 1
    else:
        expected = min(grid_cores, plan["total_tiles"], tpd.A0_KNEE_CORES)
        if spec.get("core_cap") is not None:  # A0 counterfactual row
            expected = min(expected, spec["core_cap"])
    assert plan["ncores"] == expected, (
        f"A0 violation on {name}: launched {plan['ncores']} cores, "
        f"expected {expected} (total_tiles={plan['total_tiles']}, path={plan['path']})"
    )

    if plan["path"] != "alias":
        # Lever B8 (Refinement 2) buys a THIRD CB window, so its budget is the
        # prefetch one. Either way the footprint is a constant in W, which is the
        # property `PROPERTIES["bounded_cb"]` claims.
        budget = L1_CB_BUDGET_PREFETCH_BYTES if plan["depth"] > 2 else L1_CB_BUDGET_BYTES
        assert plan["cb_bytes_per_core"] <= budget, (
            f"CB budget violation on {name}: {plan['cb_bytes_per_core']} B/core "
            f"> {budget} B (chunk_wt={plan['chunk_wt']}, depth={plan['depth']})"
        )
        if spec.get("double_buffer") is None and plan["depth"] <= 2:
            want = 2 if tpd.depth2_pays(plan["ncores"], plan["blocks_per_core"]) else 1
            assert plan["depth"] == want, (
                f"C16 gate violation on {name}: depth={plan['depth']} but the gate "
                f"wants {want} (ncores={plan['ncores']}, "
                f"blocks_per_core={plan['blocks_per_core']})"
            )
        # Refinement 2b. The fan-in path adds a staging window on top of both data
        # CBs; assert its own budget so `PROPERTIES["bounded_cb"]` still means
        # something on that path (piece_bytes is a constant in W).
        if plan["fanin_mode"] == 1:
            assert plan["cb_bytes_per_core"] <= tpd.L1_CB_BUDGET_FANIN_BYTES, (
                f"fan-in CB budget violation on {name}: {plan['cb_bytes_per_core']} B/core "
                f"> {tpd.L1_CB_BUDGET_FANIN_BYTES} B (piece={plan['piece_bytes']})"
            )
        # Refinement 2b: the shipped issue-order rotation must match the declared
        # gate on every row that does not force the lever, so narrowing the gate
        # (or a plan change that moves nt_h / chunk_wt) fails here.
        if (spec.get("levers") or {}).get("stg", 1) == 1:
            want_stg = tpd.stagger_pays(plan["ncores"], plan["nt_h"], plan["chunk_wt"])
            if plan["chunk_wt"] == 1:
                want_stg &= ~tpd.STAGGER_WRITE
            if plan["fanin_mode"] or plan["split_read"] or plan["prefetch_blocks"] == 2 or plan["stateful_read"]:
                want_stg = 0
            assert plan["stagger"] == want_stg, (
                f"stagger gate violation on {name}: stagger={plan['stagger']} but the "
                f"gate wants {want_stg} (ncores={plan['ncores']}, nt_h={plan['nt_h']}, "
                f"chunk_wt={plan['chunk_wt']})"
            )
        if plan["prefetch_blocks"] == 2:
            assert (
                plan["depth"] == tpd.PREFETCH_DEPTH
            ), f"B8 depth violation on {name}: prefetch is on but depth={plan['depth']}"
            assert (
                plan["blocks_per_core"] >= tpd.TRID_PREFETCH_MIN_BLOCKS
            ), f"B8 structural violation on {name}: only {plan['blocks_per_core']} block(s)/core"


def test_bench_tilize(device):
    """Measure; never assert correctness. The assertions here are the profiler
    producing a number (i.e. this is a profiler-enabled build) plus the two
    structural perf gates (A0 active-core count, bounded per-core CB L1)."""
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    rows = []

    for name in REGIME_NAMES:
        spec = REGIMES[name]
        # A0 counterfactual rows force a core cap through the planner's sweep hook.
        tpd.CORE_CAP_OVERRIDE = spec.get("core_cap")
        tpd.CHUNK_CAP_OVERRIDE = spec.get("chunk_cap")
        tpd.STAGGER_MOD_OVERRIDE = spec.get("stagger_mod")
        # Refinement-1c lever counterfactual rows (B13 stateful reads, C7 split
        # reader). Set before the plan is built AND before the runs, since the
        # planner reads them per call.
        levers = spec.get("levers") or {}
        for key in ("b13", "c7", "b8", "b10", "a3", "r2b", "stg"):
            os.environ[f"TILIZE_LEVER_{key.upper()}"] = str(levers.get(key, 1))
        tt_input, out_cfg = _build(device, spec)
        plan = _plan_for(device, tt_input, spec, out_cfg)
        _assert_structural_gates(name, spec, plan, grid_cores)

        elem_in = tt_input.element_size()
        bytes_read = plan["folded_h"] * plan["width"] * elem_in
        bytes_written = plan["total_tiles"] * plan["tile_out"]
        # Path B is zero-copy on BOTH sides: no DRAM traffic at all.
        traffic = 0 if plan["path"] == "alias" else bytes_read + bytes_written

        for variant, skip_dm, skip_compute in _VARIANTS:
            os.environ["TILIZE_SKIP_DM"] = skip_dm
            os.environ["TILIZE_SKIP_COMPUTE"] = skip_compute
            run_fn = lambda t=tt_input, s=spec, c=out_cfg: tilize(
                t,
                c,
                dtype=s.get("out_dtype"),
                use_multicore=s.get("multicore", True),
                use_double_buffer=s.get("double_buffer"),
            )
            ns, std = _measure_median_ns(device, run_fn)
            assert ns is not None, f"profiler produced no data for {name}/{variant}"
            rows.append(
                dict(
                    regime=name,
                    variant=variant,
                    path=plan["path"],
                    ncores=plan["ncores"],
                    chunk_wt=plan["chunk_wt"],
                    depth=plan["depth"],
                    blocks=plan["blocks_per_core"],
                    b13=plan["stateful_read"],
                    c7=plan["split_read"],
                    b8=plan["prefetch_blocks"],
                    b10=plan["vc_spread"],
                    a3=plan["bank_placement"],
                    r2b=plan["fanin_mode"],
                    stg=plan["stagger"],
                    cb_bytes=plan["cb_bytes_per_core"],
                    ns=ns,
                    cv=(std / ns * 100.0) if ns else 0.0,
                    traffic=traffic,
                )
            )

        os.environ["TILIZE_SKIP_DM"] = "0"
        os.environ["TILIZE_SKIP_COMPUTE"] = "0"
        for key in ("B13", "C7", "B8", "B10", "A3", "R2B", "STG"):
            os.environ[f"TILIZE_LEVER_{key}"] = "1"
        tpd.CORE_CAP_OVERRIDE = None
        tpd.CHUNK_CAP_OVERRIDE = None
        tpd.STAGGER_MOD_OVERRIDE = None

    arch = os.environ.get("ARCH_NAME", "unknown")
    lines = [
        "",
        "=== tilize device perf bench ===",
        f"    grid={grid.x}x{grid.y}  arch={arch}  rounds={N_ROUNDS}x{N_TRIALS} launches  ablate={ABLATE}",
        f"    A0 gate: interleaved -> ncores == min(grid_cores, total_tiles, "
        f"A0_KNEE_CORES={tpd.A0_KNEE_CORES}); sharded -> shard's own cores",
        f"    C16 gate: depth 2 iff ncores < {tpd.BANDWIDTH_KNEE_CORES} and "
        f"blk/core >= {tpd.MIN_BLOCKS_FOR_DEPTH2}",
        f"    {'regime':<34} {'variant':<11} {'path':<8} {'cores':>5} {'chk':>4} {'d':>2} "
        f"{'blk':>4} {'B13':>4} {'C7':>3} {'B8':>3} {'VC':>3} {'A3':>3} {'R2B':>4} {'STG':>4} {'cbB/core':>9} "
        f"{'ns':>10} {'cv%':>5} {'MB':>7} {'GB/s':>7}",
    ]
    for r in rows:
        gbps = (r["traffic"] / r["ns"]) if (r["traffic"] and r["ns"]) else 0.0
        lines.append(
            f"    {r['regime']:<34} {r['variant']:<11} {r['path']:<8} {r['ncores']:>5} "
            f"{r['chunk_wt']:>4} {r['depth']:>2} {r['blocks']:>4} {r['b13']:>4} {r['c7']:>3} "
            f"{r['b8']:>3} {r['b10']:>3} {r['a3']:>3} {r['r2b']:>4} {r['stg']:>4} "
            f"{r['cb_bytes']:>9} {r['ns']:>10.1f} {r['cv']:>5.1f} {r['traffic'] / 1e6:>7.2f} {gbps:>7.1f}"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    pytest.main([__file__, "-s"])

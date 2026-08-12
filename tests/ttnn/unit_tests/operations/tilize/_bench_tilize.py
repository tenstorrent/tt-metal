# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize perf bench — measurement only, NO correctness assertions.

Underscore-prefixed and deliberately NOT in `feature_spec.INPUTS`: the golden
cells are tiny (they are the correctness gate and must stay fast) and far too
small to be bandwidth-bound, so they cannot measure the thing Track A optimizes.
This file carries the grid-filling shapes instead.

    # everything (all shapes x all arms)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py

    # a subset
    TB_SHAPES=square,wide_short TB_ARMS=base,lever_b7_barrier_per_read \\
        scripts/run_safe_pytest.sh --run-all tests/.../_bench_tilize.py

Metric: `DEVICE KERNEL DURATION [ns]` read from the IN-PROCESS device profiler
(`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`), so no Tracy
CSV parsing and no host wall-clock is involved. Results are also written to
`generated/tilize_bench/<name>.json` so the changelog table can be generated
rather than typed.

**Every arm is a `levers=dict(...)` forcing arm**, so each lever's counterfactual
stays re-runnable from here instead of being an ad-hoc kernel edit — see
`ttnn/ttnn/operations/tilize/lever_ledger.json`. The `ablate_*` arms are the
/perf-measure ablation variants: they stub ONE payload while keeping every CB
reserve/push/wait/pop and every loop trip count, so their output is wrong BY
DESIGN. That is why this file asserts nothing about values (correctness lives in
the golden suite and test_tilize_debug.py, which also proves every non-stub lever
arm still computes the right answer).

Shape regimes — both grid regimes are mandatory, because a bench that measures
only the square reports healthy while a height-only split strands the wide-short
case on one core:

  square      [1,1,2048,2048]  grid-filling, several blocks/core -> per-core DRAM efficiency
  wide_short  [1,1,32,16384]   nt_h == 1                         -> does the split FILL the grid
  tall_narrow [1,1,2048,64]    n_wchunks == 1                     -> pure-height-split degenerate
  smallest    [1,1,32,64]      1 core, 1 block                    -> master.md B0: every
                               per-core-overhead lever must be counterfactualed HERE too

(The exact block/core counts move with TARGET_READ_BYTES, which is the point of
the knob; the table the bench prints reports the live numbers per shape.)
"""

import json
import os
import pathlib

# The in-process device profiler must be enabled BEFORE the device opens.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import torch

import ttnn
from ttnn.operations.tilize.tilize import _dispatch
from ttnn.operations.tilize.tilize_program_descriptor import (
    blocking,
    placement_defaults,
    plan_cores,
    plan_placement,
    shard_view,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# Device kernel duration has no warm-up transient, so this is not a "trial loop":
# WARMUP launches exist only to get the program into the cache (so the measured
# launch is not the compiling one), and TRIALS is small on purpose.
N_WARMUP = int(os.environ.get("TB_WARMUP", "2"))
N_TRIALS = int(os.environ.get("TB_TRIALS", "3"))

SHAPES = {
    "square": (1, 1, 2048, 2048),
    "wide_short": (1, 1, 32, 16384),
    "tall_narrow": (1, 1, 2048, 64),
    "smallest": (1, 1, 32, 64),
    # The smallest shape the Phase-0 op can RUN (1 core, 1 block, 1 tile). The
    # smallest shape in feature_spec.INPUTS is [1,1,30,32], but that is a
    # pad_mode="auto" cell which Phase 0 refuses, so this is its tile-aligned
    # counterpart with the identical per-core geometry (nt_h=1, Wt=1, 1 block).
    # master.md B0's per-core-overhead levers are counterfactualed here.
    "smallest_aligned": (1, 1, 32, 32),
    # Refinement 1 (A5) adds the three L1 buffer directions. `l1_to_l1` is the
    # worst case of the new axis: BOTH operands are L1-interleaved, so they spend
    # the same per-core L1 the CBs do (which is why `cb_budget_bytes` subtracts
    # them). Carried forward as a bench shape so a later phase cannot regress the
    # L1 directions while tuning the DRAM ones.
    "l1_to_l1": (1, 1, 512, 2048),
    # Refinement 2 (A3/C14) adds the SHARDED placements. Two regimes, because the
    # zero-copy lever's whole cost model is per-core:
    #   sharded_big   grid-filling same-spec HEIGHT shard on 64 cores -> the
    #                 regime where removing the NoC traffic is the whole call.
    #   sharded_small the op_requirements Refinement-6 shape: 4 cores, 8 tiles
    #                 each, where per-core fixed cost dominates (master.md B0).
    "sharded_big": (1, 1, 2048, 2048),
    "sharded_small": (1, 1, 512, 64),
    # Refinement 4 (A3c) adds the CROSS-SPEC reshard — the one topology in this
    # op where a core touches bytes another core owns. WIDTH shard in -> HEIGHT
    # shard out is its worst case: the two placements share no axis, so every
    # output core's every stick is gathered from a DIFFERENT input core, band by
    # band. Carried forward so a later phase cannot regress the gather while
    # tuning the resident or interleaved paths.
    "reshard": (1, 1, 1024, 1024),
    # The reshard's OTHER regime, and the one that decides its direction gate:
    # a WHOLE-ROW (HEIGHT-sharded) source, where a pull reads a full block row
    # (2048 B) rather than a band. `reshard` above is the banded regime, where a
    # pull reads 256 B. One shape per side of the gate, so both arms of
    # `reshard_pull` stay measurable on the geometry each one is meant to win.
    "reshard_rowwise": (1, 1, 1024, 1024),
    # Refinement 5 (P1/P2/P4/P5) adds the PADDED path. Four shapes, each chosen so
    # its padded tile grid EQUALS an existing row's, which is what makes the pad
    # body's cost readable straight off the comparison instead of against a new
    # baseline:
    #   padded_h_tail    == square's grid; H tail only -> 4 boundary blocks of 256
    #   padded_w_tail    == square's grid; W tail only -> a boundary block in EVERY
    #                       one of the 64 tile-rows (pays most often)
    #   padded_noop      == square exactly; already tile-aligned + a pad argument,
    #                       so the plan DISARMS (`pad_enabled == 0`). This row must
    #                       equal `square` to the noise band — the "a degenerate pad
    #                       is not slower" gate, measured rather than asserted.
    #   padded_row_vector== wide_short's grid; the FILL-DOMINATED regime and the pad
    #                       real models ask for (one logical row up to a tile row).
    #                       Same tiles out, 1/32 of the bytes in, and 31 of every 32
    #                       rows written by the fill instead of by the NoC.
    "padded_h_tail": (1, 1, 2046, 2048),
    "padded_w_tail": (1, 1, 2048, 2046),
    "padded_noop": (1, 1, 2048, 2048),
    "padded_row_vector": (1, 1, 1, 16384),
    # R6 ledger close-out: the GLOBAL smallest regime, i.e. the smallest shape in
    # `feature_spec.INPUTS` — [1,1,30,32], a pad_mode="auto" cell. Phase 0 refused
    # it (padding did not exist yet), which is why the per-core-overhead levers
    # were counterfactualed on its tile-aligned counterpart `smallest_aligned`
    # instead. Refinement 5 landed the padded reader, so the real smallest regime
    # is now runnable and master.md B0's rule is measurable on the shape it names:
    # 1 core, 1 block, 1 tile, and 30 of 32 rows read while 2 are filled.
    "smallest_padded": (1, 1, 30, 32),
    # Refinement 7 (A4/A5b/P3) adds the DTYPE axis, and the verifier note is that
    # the ceiling is per dtype: WT_BLOCK is a BYTE target, so the tile grid, the
    # page size and therefore the DRAM bound all move with the element size.
    # Four rows, all on the square's logical shape so the only variable is the
    # format:
    #   dtype_fp32        fp32 -> fp32, the Fp32Mode::Lossless path (which also
    #                     disables fast tilize, `tilize_helpers.inl`). 4x the
    #                     bytes of `square`, WT_BLOCK 8.
    #   dtype_uint8       uint8 -> uint8, the 32-bit-DEST path. 1/2 the bytes,
    #                     WT_BLOCK 32; W=2048 keeps the read ALIGNED.
    #   dtype_uint8_narrow  the A5b staged reader's own regime (Wt == 1, so the
    #                     per-stick destination is off the 64 B grid and every
    #                     block is staged + compacted). Its cost has no off-arm
    #                     because the direct read is not legal, so it is read
    #                     against `dtype_uint8`'s per-byte rate.
    #   dtype_bf16_to_bf8b  the block-float OUTPUT, where the packer choice is a
    #                     real perf knob (`bfp8_precise`).
    "dtype_fp32": (1, 1, 2048, 2048),
    "dtype_uint8": (1, 1, 2048, 2048),
    "dtype_uint8_narrow": (1, 1, 65536, 32),
    "dtype_bf16_to_bf8b": (1, 1, 2048, 2048),
    # Refinement 8 (T1/T2) adds the TILE GEOMETRY axis. All four rows sit on the
    # square's logical shape so the only variable is the tile, and the block
    # geometry the tile height implies is readable straight off `square`:
    #   tile_16 / tile_1  the TINY-TILE path. Same bytes, same Wt, but `tile_h`
    #                     sticks per block instead of 32 -> 2x / 32x the block
    #                     COUNT at the same block width, i.e. the regime where a
    #                     per-block fixed cost is multiplied. `tile_1` also takes
    #                     the `copy`-chain compute branch rather than `tilize`.
    #   retile_shrink     TILE(32) in -> tile 8 out. The staged face-walk reader,
    #                     in its OVER-READ direction: a block of 8 output sticks
    #                     stages whole 32-row source tiles, so it reads 4x the
    #                     bytes. op_design §8.4 rules Track T correctness-gated,
    #                     so this row is a WATCHED number, not a ceiling target.
    #   retile_grow       TILE(8) in -> tile 32 out. The mirror direction, where
    #                     the same reader reads each source tile exactly once.
    # A smaller logical shape than the square's, because the face walk is a
    # per-datum L1 copy and the row is here to be re-measurable, not to be fast.
    "tile_16": (1, 1, 2048, 2048),
    "tile_1": (1, 1, 2048, 2048),
    "retile_shrink": (1, 1, 1024, 1024),
    "retile_grow": (1, 1, 1024, 1024),
}

# R8: per-shape TILE GEOMETRY — `(in_tile_height | None, out_tile_height)`. One
# source of truth in the same shape as `_MEM_BY_SHAPE` / `_PAD_BY_SHAPE` /
# `_DTYPE_BY_SHAPE`. `None` on the input side is a ROW_MAJOR input (every shape
# absent here); an int is the RETILE path, where the bench has to build the
# input already tiled at that height.
_TILE_BY_SHAPE = {
    "tile_16": (None, 16),
    "tile_1": (None, 1),
    "retile_shrink": (32, 8),
    "retile_grow": (8, 32),
}


def _tiles_for(shape_name):
    return _TILE_BY_SHAPE.get(shape_name, (None, 32))


# R7: per-shape (input dtype, output dtype). One source of truth, in the same
# shape as `_MEM_BY_SHAPE` / `_PAD_BY_SHAPE`; a shape absent here is the bf16
# identity every earlier phase measured.
_DTYPE_BY_SHAPE = {
    "dtype_fp32": (ttnn.float32, ttnn.float32),
    "dtype_uint8": (ttnn.uint8, ttnn.uint8),
    "dtype_uint8_narrow": (ttnn.uint8, ttnn.uint8),
    "dtype_bf16_to_bf8b": (ttnn.bfloat16, ttnn.bfloat8_b),
}

# Bytes per L1 datum, for the GB/s column and the block geometry in the header.
# `bfloat8_b` is block-float (no fixed datum size); its TILE is 1088 B for 1024
# datums, so 1.0625 is the honest per-datum figure.
_ELEM_BYTES = {
    ttnn.bfloat16: 2,
    ttnn.float32: 4,
    ttnn.uint32: 4,
    ttnn.uint16: 2,
    ttnn.int32: 4,
    ttnn.uint8: 1,
    ttnn.bfloat8_b: 1.0625,
}


def _dtypes_for(shape_name):
    return _DTYPE_BY_SHAPE.get(shape_name, (ttnn.bfloat16, ttnn.bfloat16))


def _height_shard(grid_end, shard_shape):
    """Same-spec L1 HEIGHT shard (both sides) — the zero-copy placement."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*grid_end))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


# 2048 rows over an 8x8 grid -> 32 rows/core (one tile-row, 64 tiles wide), and
# 512 rows over 4 cores -> 128 rows/core (4 tile-rows, 2 tiles wide).
_SHARD_BIG = _height_shard((7, 7), (32, 2048))
_SHARD_SMALL = _height_shard((3, 0), (128, 64))

# R4: the cross-spec pair. 1024 rows over 8 cores -> a (128,1024) HEIGHT shard
# out; 1024 columns over the same 8 cores -> a (1024,128) WIDTH shard in, i.e.
# 8 source bands per row. No core's input shard overlaps its output shard in
# more than one block, so the whole call is gather.
_RESHARD_IN = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        (1024, 128),
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
_RESHARD_OUT = _height_shard((7, 0), (128, 1024))

# The whole-row regime: 8 HEIGHT shards in -> 4 HEIGHT shards out (a merge), so
# the source pages are full rows (n_bands == 1) and a pull reads whole block rows.
_RESHARD_ROW_IN = _height_shard((3, 0), (256, 1024))
_RESHARD_ROW_OUT = _height_shard((7, 0), (128, 1024))

# Per-shape memory placement; DRAM interleaved on both sides unless named here.
# ONE source of truth for a shape's placement — `_bench_input` and the `_dispatch`
# call both read it.
_MEM_BY_SHAPE = {
    "l1_to_l1": (ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    "sharded_big": (_SHARD_BIG, _SHARD_BIG),
    "sharded_small": (_SHARD_SMALL, _SHARD_SMALL),
    "reshard": (_RESHARD_IN, _RESHARD_OUT),
    "reshard_rowwise": (_RESHARD_ROW_IN, _RESHARD_ROW_OUT),
}


def _mem_for(shape_name):
    return _MEM_BY_SHAPE.get(shape_name, (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG))


# R5: per-shape PAD request, same shape as `_MEM_BY_SHAPE` — one source of truth,
# read by the `_dispatch` call below. A shape absent here asks for no padding at
# all and is the Track A path verbatim (`pad_plan` returns None).
_PAD_BY_SHAPE = {
    "padded_h_tail": dict(pad_value=-18.5),
    "padded_w_tail": dict(pad_value=-18.5),
    "padded_noop": dict(pad_value=-18.5),
    "padded_row_vector": dict(pad_value=-18.5),
    "smallest_padded": dict(pad_value=-18.5),
}


def _pad_for(shape_name):
    return _PAD_BY_SHAPE.get(shape_name, {})


# arm name -> the kwargs handed to `_dispatch`. `base` is the shipped
# configuration (`levers=dict()` == DEFAULT_LEVERS); every other arm flips
# exactly ONE lever off (or stubs one payload for ablation). Keeping every arm in
# the `levers=dict(<knob>=<value>)` shape is what makes each counterfactual
# re-runnable — `eval/verify_levers.py` scans this file for exactly that form.
ARMS = {
    # ---- baseline -------------------------------------------------------
    "base": dict(levers=dict()),
    # ---- distribution levers (A0 / A1) ----------------------------------
    "base_singlecore": dict(levers=dict(multicore=0)),  # A0 off-arm
    "lever_a1_width_split_off": dict(levers=dict(width_split=0)),
    "lever_a1_row_wise_off": dict(levers=dict(row_wise=0)),
    # ---- transaction-shape levers (B5 / B6 / B7 / B9) -------------------
    "lever_b6_read_128": dict(levers=dict(target_read_bytes=128)),
    "lever_b6_read_256": dict(levers=dict(target_read_bytes=256)),
    "lever_b6_read_512": dict(levers=dict(target_read_bytes=512)),
    "lever_b6_read_2048": dict(levers=dict(target_read_bytes=2048)),
    "lever_b6_read_4096": dict(levers=dict(target_read_bytes=4096)),
    "lever_b7_barrier_per_read": dict(levers=dict(barrier_per_block=0)),
    # ---- R3 placement levers: pipeline depth + core spread ------------------
    # Both knobs ship at a REGIME-selected default (`placement_defaults`: the
    # pipeline cap on the all-L1 path, the spread on the all-DRAM path), because
    # they measure with OPPOSITE signs on those two paths. So each lever needs BOTH
    # an off-arm (on the path where it ships on) and a force-arm (on the path where
    # it is gated off) — the force-arms are the evidence FOR the gate, and an
    # explicit value here always overrides the regime default.
    #
    # pipeline depth: blocks per core == the number of read/compute/write stages a
    # core can overlap. Ships on l1_to_l1 (1.35x); forced on wide_short it trades
    # 32 cores for 16 and is a wash.
    "lever_r3_pipeline_off": dict(levers=dict(min_blocks_per_core=1)),
    "lever_r3_pipeline_force2": dict(levers=dict(min_blocks_per_core=2)),
    "lever_r3_pipeline_force3": dict(levers=dict(min_blocks_per_core=3)),
    "lever_r3_pipeline_force4": dict(levers=dict(min_blocks_per_core=4)),
    # core spread: which cores, when active_cores < grid_cores. Ships on the DRAM
    # path (wide_short 1.05x); forced on l1_to_l1 it costs 1.08x, which is the gate.
    "lever_r3_spread_off": dict(levers=dict(spread_cores=0)),
    "lever_r3_spread_force": dict(levers=dict(spread_cores=1)),
    "lever_r3_spread_off_pipeline_force2": dict(levers=dict(spread_cores=0, min_blocks_per_core=2)),
    # the co-tuning corners the refinement asks for: pipeline depth x block size
    # (`target_read_bytes` -> WT_BLOCK -> block count -> cores).
    "lever_r3_pipeline_force2_read512": dict(levers=dict(min_blocks_per_core=2, target_read_bytes=512)),
    "lever_r3_pipeline_force2_read2048": dict(levers=dict(min_blocks_per_core=2, target_read_bytes=2048)),
    "lever_r3_gridfill_read512_off": dict(levers=dict(min_blocks_per_core=1, target_read_bytes=512)),
    # read-issue stagger (master.md A3): rotate each core's read order by its own
    # block index, so a fleet that shares source pages does not queue on one bank.
    "lever_r3_stagger_off": dict(levers=dict(stagger_reads=0)),
    "lever_r3_stagger_force": dict(levers=dict(stagger_reads=1)),
    # R6 (master.md B13 + D21): configure the NoC command buffer once per source
    # BANK and issue the block's remaining reads with `with_state` — three
    # command-buffer register writes instead of six. A per-ISSUE-cost lever, so
    # per master.md B0 its off-arm is measured on the smallest regimes first.
    "lever_r6_stateful_off": dict(levers=dict(stateful_reads=0)),
    "lever_r6_stateful_force": dict(levers=dict(stateful_reads=1)),
    # R6 (master.md D21): derive the addresses inside a bank group by addition
    # instead of paying `TensorAccessor`'s two divides-by-7 per stick.
    "lever_r6_addrgen_off": dict(levers=dict(fast_addrgen=0)),
    "lever_r6_addrgen_force": dict(levers=dict(fast_addrgen=1)),
    "lever_r6_addrgen_and_stateful": dict(levers=dict(fast_addrgen=1, stateful_reads=1)),
    # R6 (master.md C14, SECOND degree): fold the resident path's two dataflow
    # kernels into compute. Both arms are forcing arms because the choice is a
    # per-regime measurement, not a default.
    # R6: the tilize LLK teardown, which is per-CALL fixed cost — the term the
    # low-work regimes are made of.
    "lever_r6_no_uninit": dict(levers=dict(tilize_uninit=0)),
    "lever_r6_uninit_force": dict(levers=dict(tilize_uninit=1)),
    # R6: one input wait per CALL instead of per block, on the resident path
    # (where the whole shard is already in the CB before compute starts).
    "lever_r6_wait_upfront_off": dict(levers=dict(wait_upfront=0)),
    "lever_r6_fold_off": dict(levers=dict(fold_resident=0)),
    "lever_r6_fold_force": dict(levers=dict(fold_resident=1)),
    "lever_b5_face_writes": dict(levers=dict(coalesce_writes=0)),
    "lever_b9_noc_swap": dict(levers=dict(noc_split=0)),
    # ---- buffering lever (C16) ------------------------------------------
    "lever_c16_depth1": dict(levers=dict(double_buffer=0)),
    # ---- placement lever (A2 / C14): zero-copy OFF ----------------------
    # The sharded shapes' counterfactual — consume the resident shard through a
    # TensorAccessor instead of aliasing the CB onto it (i.e. the interleaved
    # path merely TOLERATING the layout). A no-op on the interleaved shapes.
    "lever_c14_force_streamed": dict(levers=dict(force_streamed=1)),
    # R4 (A3c): the cross-spec reshard's DIRECTION. `pull` is op_design §4.3's
    # contract (the output shard is resident and gathers); `push` is its mirror
    # (the input shard is resident and scatters whole tile pages). Both arms are
    # forcing arms so the choice is measured on any cross-spec geometry.
    "lever_r4_reshard_push": dict(levers=dict(reshard_pull=0)),
    "lever_r4_reshard_pull": dict(levers=dict(reshard_pull=1)),
    # R7 (A4/A5b) — the two numeric-format knobs, each with its off-arm. Both are
    # CORRECTNESS levers on their own dtype (the off-arm is numerically wrong
    # there, which is exactly what makes it worth knowing what correctness
    # costs), and both are no-ops on every other dtype.
    "lever_r7_fp32_fast": dict(levers=dict(fp32_lossless=0)),
    "lever_r7_dest16": dict(levers=dict(fp32_dest_acc_8bit=0)),
    # R7 — the bfp8 packer. The ONLY one of the three that is a pure perf knob:
    # the fast packer truncates the block-float mantissas, the precise one rounds
    # and costs an extra pack pass. Shipped gated on an fp32 INPUT
    # (/numeric-formats-metal 1.7), so both arms have to stay measurable.
    "lever_r7_bfp8_precise": dict(levers=dict(bfp8_precise=1)),
    "lever_r7_bfp8_fast": dict(levers=dict(bfp8_precise=0)),
    # R7 (master.md B11) — the narrow-stick staged read's off-arm. Numerically
    # WRONG on the shapes that arm it (that is the defect), measured only for
    # what alignment-correctness costs.
    "lever_b11_stage_off": dict(levers=dict(stage_reads=0)),
    # R9 (master.md B10) — the per-core unicast VC, on BOTH NoC halves (the
    # reader's request VC and the writer's write VC, one number per core). A
    # per-core-overhead lever (one extra command-buffer register write per
    # kernel), so per master.md B0 both arms are measured on the smallest
    # regimes as well as on the square it targets.
    # The mask is per-half (1 = reader, 2 = writer, 3 = both) because the two
    # halves do NOT behave alike — see the R9 table.
    "lever_r9_vc_force": dict(levers=dict(per_core_vc=3)),
    "lever_r9_vc_reader": dict(levers=dict(per_core_vc=1)),
    "lever_r9_vc_writer": dict(levers=dict(per_core_vc=2)),
    "lever_r9_vc_off": dict(levers=dict(per_core_vc=0)),
    # R9 (master.md A3, the residual degree) — the block -> core MAPPING order.
    # Row-major makes a core's consecutive blocks walk ACROSS one tile-row, so
    # it re-reads the same source pages (same DRAM banks) at successive byte
    # offsets instead of touching `nb * tile_h` distinct pages. This is the only
    # placement freedom left once the grid is full (`spread_cores` is inert at
    # 110 of 110 cores).
    "lever_r9_block_row": dict(levers=dict(block_order=1)),
    "lever_r9_block_col": dict(levers=dict(block_order=0)),
    # The pair — levers compose, and a flat single-lever measurement is usually
    # a co-binding stage rather than a dead lever.
    "lever_r9_vc_and_block_row": dict(levers=dict(per_core_vc=3, block_order=1)),
    # ---- ablation arms (classification; output wrong by design) ---------
    "ablate_compute": dict(levers=dict(stub_compute=1)),
    "ablate_read": dict(levers=dict(stub_read=1)),
    "ablate_write": dict(levers=dict(stub_write=1)),
    "ablate_read_compute": dict(levers=dict(stub_read=1, stub_compute=1)),
    "ablate_all": dict(levers=dict(stub_read=1, stub_compute=1, stub_write=1)),
}

_OUT_DIR = pathlib.Path("generated/tilize_bench")


def _selected(env_name, universe):
    raw = os.environ.get(env_name)
    if not raw:
        return list(universe)
    names = [n.strip() for n in raw.split(",") if n.strip()]
    for name in names:
        if name not in universe:
            raise ValueError(f"{env_name}: unknown entry {name!r}; known: {list(universe)}")
    return names


def _bench_input(shape, device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, in_tile_height=None):
    torch.manual_seed(0)
    if dtype == ttnn.uint8:
        source = torch.randint(0, 256, shape, dtype=torch.uint8)
    elif dtype in (ttnn.uint32, ttnn.uint16, ttnn.int32):
        source = torch.randint(0, 100, shape, dtype=torch.int32)
    elif dtype == ttnn.float32:
        source = torch.randn(shape, dtype=torch.float32)
    else:
        source = torch.randn(shape).bfloat16()
    # R8 (T2): the retile rows need the input ALREADY tiled, at its own tile
    # height. Every other row is the ROW_MAJOR input every earlier phase measured.
    if in_tile_height is not None:
        return ttnn.from_torch(
            source,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
            tile=ttnn.Tile([in_tile_height, 32]),
        )
    return ttnn.from_torch(
        source,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _measure_ns(device, run_fn):
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush the warm-up window
    for _ in range(N_TRIALS):
        run_fn()
    ttnn.synchronize_device(device)
    total = _read_kernel_ns(device)
    return None if total is None else total / N_TRIALS


def test_bench(device):
    """Measure every selected (shape, arm) and print the table. Perf is evidence,
    never pass/fail — the only assertion is that the profiler produced numbers."""
    shape_names = _selected("TB_SHAPES", SHAPES)
    arm_names = _selected("TB_ARMS", ARMS)
    grid = device.compute_with_storage_grid_size()

    results = {}
    for shape_name in shape_names:
        shape = SHAPES[shape_name]
        in_mem, out_mem = _mem_for(shape_name)
        in_dtype, out_dtype = _dtypes_for(shape_name)
        in_tile_h, out_tile_h = _tiles_for(shape_name)
        tt_input = _bench_input(shape, device, dtype=in_dtype, memory_config=in_mem, in_tile_height=in_tile_h)
        pad_kwargs = _pad_for(shape_name)
        if out_dtype != in_dtype:
            pad_kwargs = dict(pad_kwargs, dtype=out_dtype)
        # R8: `tile=` is passed only for a non-default geometry, so every earlier
        # phase's row keeps exercising the exact call shape it was measured with.
        if out_tile_h != 32 or in_tile_h is not None:
            pad_kwargs = dict(pad_kwargs, tile=ttnn.Tile([out_tile_h, 32]))
        for arm in arm_names:
            kwargs = dict(ARMS[arm], **pad_kwargs)
            run_fn = lambda kw=kwargs, t=tt_input, om=out_mem: _dispatch(t, om, use_multicore=True, **kw)
            ns = _measure_ns(device, run_fn)
            assert ns is not None, f"profiler produced no data for {shape_name}/{arm}"
            results[(shape_name, arm)] = ns

    # ---- report ----
    lines = [
        "",
        "=== tilize bench — DEVICE KERNEL DURATION [ns], "
        f"grid={grid.x}x{grid.y}={grid.x * grid.y}, {N_TRIALS} launches averaged ===",
    ]
    payload = {}
    for shape_name in shape_names:
        shape = SHAPES[shape_name]
        in_dtype, out_dtype = _dtypes_for(shape_name)
        elem_bytes = _ELEM_BYTES[in_dtype]
        # R8: the reported block geometry is the one the row actually ran with,
        # so the tile height comes from `_TILE_BY_SHAPE` rather than a literal 32.
        row_tile_h = _tiles_for(shape_name)[1]
        blk = blocking(list(shape), row_tile_h, int(elem_bytes))
        in_mem, out_mem = _mem_for(shape_name)
        if in_mem.is_sharded() or out_mem.is_sharded():
            # A shard's cores are fixed by its spec (master.md A2), and the shard
            # hands you the block width — so neither comes from `plan_cores`, and
            # on a cross-spec reshard it is the RESIDENT side that hands them
            # over (the output, under R4's pull topology), not the input.
            plan = plan_placement(
                shape=list(shape),
                tile_height=row_tile_h,
                in_memory_config=in_mem,
                out_memory_config=out_mem,
                Wt=blk["Wt"],
                nt_h=blk["nt_h"],
                in_tile_bytes=row_tile_h * 32 * int(elem_bytes),
                out_tile_bytes=row_tile_h * 32 * int(elem_bytes),
            )
            side_mem = in_mem if plan["sharded_side"] == "in" else out_mem
            nt_h_shard = plan["shard"]["nt_h_shard"]
            cores = list(range(shard_view(side_mem)[0].num_cores()))
            blk = dict(blk, wt_block=plan["wt_block"], total_blocks=len(cores) * nt_h_shard)
            per_core = [nt_h_shard]
        else:
            # Report the SHIPPED geometry, so the header's core count and pipeline
            # depth are the ones `base` actually ran with (R3's cap included).
            gate = placement_defaults(in_mem, out_mem)
            cores, _all_cores, per_core = plan_cores(
                device,
                blk["total_blocks"],
                use_multicore=True,
                min_blocks_per_core=gate["min_blocks_per_core"],
                spread_cores=bool(gate["spread_cores"]),
            )
        # read (input format) + write (output format) — they differ on a cast.
        datums = torch.tensor(shape).prod().item()
        total_bytes = int(datums * (elem_bytes + _ELEM_BYTES[out_dtype]))
        base = results.get((shape_name, "base"))
        lines += [
            "",
            f"  {shape_name} {tuple(shape)}: nt_h={blk['nt_h']} Wt={blk['Wt']} "
            f"WT_BLOCK={blk['wt_block']} n_wchunks={blk['n_wchunks']} "
            f"blocks={blk['total_blocks']} cores={len(cores)} "
            f"blocks/core={max(per_core) if per_core else 0} dram_bytes={total_bytes}",
            f"    {'arm':<28} {'ns':>12} {'vs base':>9} {'GB/s':>8}",
        ]
        for arm in arm_names:
            ns = results[(shape_name, arm)]
            ratio = f"{ns / base:.3f}x" if base else "-"
            gbps = total_bytes / ns if ns else 0.0
            lines.append(f"    {arm:<28} {ns:>12.1f} {ratio:>9} {gbps:>8.1f}")
            payload[f"{shape_name}/{arm}"] = {
                "ns": ns,
                "vs_base": (ns / base) if base else None,
                "gbps": gbps,
                "shape": list(shape),
                "cores": len(cores),
                "blocks": blk["total_blocks"],
                "wt_block": blk["wt_block"],
                "dtype": str(in_dtype),
                "output_dtype": str(out_dtype),
            }
    print("\n".join(lines))

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = os.environ.get("TB_OUT", "latest")
    (_OUT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"    -> {_OUT_DIR / (name + '.json')}")

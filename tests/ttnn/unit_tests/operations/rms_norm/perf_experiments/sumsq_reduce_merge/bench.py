# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED BAKE-OFF (idea I6): collapse rms_norm's TWO statistics phases into ONE.

Scope = ONLY the per-tile-row "sum of squares -> column-0-valid partial" stage of
rms_norm. Everything else (DRAM movement, the cross-core combine, rsqrt, the apply
pass, gamma) is absent: the input block is already resident in L1 as a single-core
shard, and the output is a single-core shard. So the measured delta is attributable
to this stage alone.

WHAT THE STAGE MUST PRODUCE
    out[r] (column-0-valid tile r) = (1 / W_true) * sum over the core's hidden slice
    of x[row, c]^2, per row of the tile-row r. The 1/W_true divisor rides here
    (rms_norm Refinement 5 folded it into the reduce scaler), so each per-core
    partial is already a share of the MEAN.

VARIANTS (all under the SAME user precision contract: bf16 in / bf16 stat out,
MathFidelity.HiFi2, fp32_dest_acc_en=False -- nothing here tunes those)

  baseline        The op's CURRENT approach, reconstructed faithfully from
                  rms_norm_compute.cpp:
                    PHASE A  eltwise_chain BinaryFpu<Mul, DestAccumulation::PerRow>
                             over grid(rows_t, cols) with TileOffset::Strided (and
                             the caller-managed (None,None) CB policies Strided
                             requires) -> PACKS one tile per tile-row into cb_stat_sq,
                             one stat COLUMN per hidden chunk, plus the masked
                             (x*wmask)^2 tail column when the core owns a ragged tile.
                    PHASE B  ckl::reduce<SUM, REDUCE_ROW, cb_stat_sq, cb_scaler,
                             cb_stat_partial, BulkWaitBulkPop> over (rows_t, nc, 1)
                             -- unpacks those tiles back in and folds the WITHIN-TILE
                             columns; 1/W_true rides in the scaler.
  baseline_norecfg  identical, but the phase-boundary DATA-FORMAT RECONFIG is off
                  (DataFormatReconfig::Disabled on every chain operand +
                  ReduceDataFormatReconfigMode::NONE). Splits "the boundary costs"
                  from "the L1 round trip costs". NOT a candidate for graduation on
                  its own (see the note in the report) -- a measurement instrument.
  baseline_accadd PHASE B swapped for the library opt-in
                  ReduceAlgorithm::AccumulateViaAdd (via reduce_mean, whose 1/N is the
                  caller's -- we pass W_true). Answers "can the opt-in consume the
                  DEST-resident accumulator?" -- it cannot (it reads its inputs from
                  L1 through add_tiles), so this only replaces the FPU matmul-reduce
                  with the SFPU finalize on the SAME L1 round trip.
  merged          THE CANDIDATE. One phase: the x^2 accumulator NEVER leaves DEST.
                  Per tile-row: acc_to_dest mul_tiles over the whole hidden slice ->
                  (ragged tail: masked bcast-mul into a transient DEST slot, SFPU
                  square, SFPU add into the accumulator) -> sfpu_reduce<SUM,
                  REDUCE_ROW> collapses the within-tile columns IN DEST ->
                  mul_unary_tile(1/W_true) -> ONE pack straight into cb_stat_partial.
                  cb_stat_sq does not exist. Neither does the second phase's init or
                  format reconfig, nor the per-chunk stat-column bookkeeping (a DEST
                  accumulator spans the chunks for free).
  merged_hoist    same, with the FPU MOP init and the SFPU reduce-macro load hoisted
                  out of the tile-row loop (the reduce library hoists its
                  sfpu_reduce_init out of its per-output loop the same way).
  merged_cvalid   merged_hoist + the 1/W_true multiply scoped to VectorMode::C (the
                  collapsed value is column-0-valid, so Face0+Face2 is all that has
                  to be touched). The BEST merged form measured; `mul_unary_tile`
                  hardcodes VectorMode::RC, which is 2x the vector ops for the same
                  result at the same precision.
  merged_noscale  ABLATION, not an option: merged_hoist with the 1/W_true multiply
                  DELETED (answer wrong by a uniform factor W_true), to price that one
                  SFPU op. Same category as the op's own RMSN_ABLATE_* switches.
  baseline_onechain  ONE eltwise_chain over the whole hidden slice instead of one per
                  hidden CHUNK, so nc collapses to 1 (+ the tail column) and the FPU
                  reduce is untouched. Pure helper code. Isolates how much of the
                  merged form's chunked-regime win is "no per-chunk stat columns"
                  rather than "no L1 round trip". Requires a single-stride row-major
                  resident block (TILE input / pinned shard); a ROW_MAJOR input is
                  staged CHUNK-major by tilize<CB_CHUNK_TILES>, whose per-row tile
                  addresses are not one stride, so it is inexpressible there.

Correctness is the only pass/fail. Every variant is compared against a torch
reference and reported with BOTH pcc and the got/true ratio median (a lost scaler or
a lost mantissa is a uniform scale shift that pcc hides).

MEASURED RESULT (blackhole_p150b, one Tensix core, bf16/HiFi2/fp32_dest_acc_en=False):
the merge is a REGRESSION. Per tile-row the SFPU within-tile column collapse costs
~115 ns MORE than the (pack + unpack + FPU matmul-reduce) round trip it deletes, and
the merged form additionally has to pay 60-107 ns for the 1/W_true multiply that the
reduce's SCALER carries for free; against that the merge saves ~75-90 ns of ONE phase
boundary PER BLOCK. Break-even needs fewer than one tile-row per block, so there is
none. See the numbers in `harness.py`'s docstring recipe output.
"""

from __future__ import annotations

import struct

import ttnn

TILE = 32

CB_IN = 0
CB_SCALER = 2
CB_WMASK = 3
CB_STAT_SQ = 5
CB_OUT = 7

V_BASELINE = 0
V_NORECFG = 1
V_ACCADD = 2
V_MERGED = 3
V_MERGED_HOIST = 4
V_MERGED_CVALID = 5
V_MERGED_NOSCALE = 6
V_ONECHAIN = 7

VARIANTS = {
    "baseline": V_BASELINE,
    "baseline_norecfg": V_NORECFG,
    "baseline_accadd": V_ACCADD,
    "baseline_onechain": V_ONECHAIN,
    "merged": V_MERGED,
    "merged_hoist": V_MERGED_HOIST,
    "merged_cvalid": V_MERGED_CVALID,
    "merged_noscale": V_MERGED_NOSCALE,
}
BASELINE = "baseline"
_MERGED = {"merged", "merged_hoist", "merged_cvalid", "merged_noscale"}
# `merged_noscale` is an ABLATION, not an option: it drops the 1/W_true multiply
# (so its answer is wrong by a uniform factor W_true) while keeping every other
# instruction, to price that one SFPU op. Same category as the op's own
# RMSN_ABLATE_* switches.
ABLATIONS = {"merged_noscale", "baseline_norecfg"}


# =============================================================================
# Compute kernel. One source, `variant` is a compile-time arg.
#
# RAW LLK / RAW COMPUTE-API JUSTIFICATION (the `merged` variants only)
#   Helper bypassed: the pair `ckl::eltwise_chain(... DestAccumulation::PerRow ...)`
#   + `ckl::reduce<SUM, REDUCE_ROW, ...>`.
#   What the helpers cannot express: a per-tile-row DEST accumulator that is
#   FINALIZED IN DEST. `eltwise_chain` runs every non-pack chain element once per
#   tile of the (Ht x Wt) walk (eltwise_chain.inl: elem_apply_compute sits inside the
#   inner `wt` loop, only elem_apply_pack is hoisted to the row boundary), so an
#   appended SFPU column-collapse would fire on every hidden tile and destroy the
#   running accumulator instead of finalizing it -- there is no "once per row, after
#   the accumulation, before the pack" hook. And `ckl::reduce`, including the
#   `ReduceAlgorithm::AccumulateViaAdd` opt-in, only ever reads its reduce-dim tiles
#   from an L1 CB (reduce_helpers_compute.inl: add_tiles / copy_tile against
#   input_dfb_id) -- it cannot be handed an already-DEST-resident accumulator, and it
#   sums tiles rather than squaring them. So the merged phase is a MISSING BLOCK
#   OPERATION, and the sequence below is what the helper would have to grow.
#   The primitives used are the same ones both helpers use internally:
#   mul_tiles_init(acc_to_dest) + mul_tiles (what BinaryFpu<Mul, PerRow> emits),
#   sfpu_reduce_init / sfpu_reduce / mul_unary_tile (what reduce's AccumulateViaAdd
#   finalize emits).
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_wmask = 3;
constexpr uint32_t cb_stat_sq = 5;
constexpr uint32_t cb_out = 7;

constexpr uint32_t V_BASELINE = 0;
constexpr uint32_t V_NORECFG = 1;
constexpr uint32_t V_ACCADD = 2;
constexpr uint32_t V_MERGED = 3;
constexpr uint32_t V_MERGED_HOIST = 4;
constexpr uint32_t V_MERGED_CVALID = 5;
constexpr uint32_t V_MERGED_NOSCALE = 6;
constexpr uint32_t V_ONECHAIN = 7;
}  // namespace

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    // Row stride of the resident input block, exactly the op's CB_W_TILES.
    constexpr uint32_t CB_W_TILES = get_compile_time_arg_val(1);
    // The op's CB_CHUNK_TILES: the hidden-axis chunk width of the statistics walk.
    constexpr uint32_t CB_CHUNK_TILES = get_compile_time_arg_val(2);
    constexpr uint32_t ITERS = get_compile_time_arg_val(3);
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(4);  // float bits of 1/W_true
    constexpr uint32_t W_TRUE = get_compile_time_arg_val(5);

    const uint32_t rows_t = get_arg_val<uint32_t>(0);
    const uint32_t core_w = get_arg_val<uint32_t>(1);
    const uint32_t has_tail = get_arg_val<uint32_t>(2);

    constexpr bool MERGED = (VARIANT == V_MERGED) || (VARIANT == V_MERGED_HOIST) ||
                            (VARIANT == V_MERGED_CVALID) || (VARIANT == V_MERGED_NOSCALE);
    constexpr bool HOIST = (VARIANT != V_MERGED);  // every merged form but the safe per-row-init one
    // Column-0-valid 1/W multiply: after the within-tile REDUCE_ROW collapse only
    // column 0 is ever read (the apply consumes the stat as an OperandKind::Col
    // broadcast), and VectorMode::C walks Face0+Face2 -- the half of the tile that
    // holds column 0 -- so it is HALF the SFPU vector ops of the RC default that
    // `mul_unary_tile` hardcodes.
    constexpr bool CVALID_SCALE = (VARIANT == V_MERGED_CVALID);
    constexpr bool NO_SCALE = (VARIANT == V_MERGED_NOSCALE);  // ABLATION: prices the 1/W SFPU op
    // ONE eltwise_chain over the WHOLE hidden slice instead of one per chunk: the
    // DEST accumulator already spans the chunks, so the per-chunk stat COLUMN only
    // exists to give each chain call somewhere to pack. Keeps the cheap FPU
    // matmul-reduce. Only expressible for a ROW-MAJOR-strided resident block (a
    // TILE input or a pinned shard); a ROW_MAJOR input is staged CHUNK-major, whose
    // per-row tile addresses are not a single stride.
    constexpr bool ONECHAIN = (VARIANT == V_ONECHAIN);
    // Per-phase zones only in the ITERS == 1 pass: 2 markers per zone EXECUTION and
    // 250 per RISC, so a per-phase zone inside a 64-iteration loop would silently
    // truncate the profile (perf_instrumentation.hpp marker-budget note).
    constexpr bool PHASE_ZONES = (ITERS == 1);

    constexpr auto RECFG =
        (VARIANT == V_NORECFG) ? ckl::DataFormatReconfig::Disabled : ckl::DataFormatReconfig::Enabled;
    constexpr auto RMODE = (VARIANT == V_NORECFG) ? ckl::ReduceDataFormatReconfigMode::NONE
                                                  : ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;

    {
        MaybeDeviceZoneScope("cp_hw_startup");
        if constexpr (MERGED) {
            // ONE pack target for the whole kernel (cb_stat_sq is gone), so the pack
            // side is programmed once at boot and never reconfigured again.
            compute_kernel_hw_startup(cb_in, cb_in, cb_out);
        } else {
            compute_kernel_hw_startup(cb_in, cb_scaler, cb_stat_sq);
        }
    }

    // The input block is a resident shard: mark it present once, never pop it (the
    // op's cb_input_tiles lifetime across the whole block).
    const uint32_t in_block_pages = rows_t * CB_W_TILES;
    cb_reserve_back(cb_in, in_block_pages);
    cb_push_back(cb_in, in_block_pages);
    cb_wait_front(cb_in, in_block_pages);
    if (has_tail) {
        cb_wait_front(cb_wmask, 1);
    }

    // ---- operand specs, shared by both bulk and tail chains of PHASE A ----
    constexpr auto in_spec = ckl::input(
        cb_in,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::OperandKind::Block,
        RECFG,
        ckl::TileOffset::Strided);
    constexpr auto mask_spec =
        ckl::input(cb_wmask, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Scalar, RECFG);

    // ONECHAIN collapses the chunk walk into one; every other variant reproduces the
    // op's CB_CHUNK_TILES-wide walk exactly.
    const uint32_t chunk_w = ONECHAIN ? (core_w - has_tail > 0 ? core_w - has_tail : 1u) : CB_CHUNK_TILES;

    auto phase_a = [&]() {
        const uint32_t c_full = core_w - has_tail;
        const uint32_t bulk_cols = (c_full + chunk_w - 1) / chunk_w;
        const uint32_t nc = bulk_cols + has_tail;
        const uint32_t tail_col = bulk_cols;
        cb_reserve_back(cb_stat_sq, rows_t * nc);
        for (uint32_t k = 0; k < bulk_cols; ++k) {
            const uint32_t chunk_base = k * chunk_w;
            const uint32_t cols = (c_full - chunk_base < chunk_w) ? (c_full - chunk_base) : chunk_w;
            const ckl::StridedTileRange src{chunk_base, CB_W_TILES};
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(rows_t, cols),
                ckl::BinaryFpu<
                    in_spec,
                    in_spec,
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::None,
                    ckl::Dst::D0,
                    ckl::DestAccumulation::PerRow>{src, src},
                ckl::PackTile<ckl::output(
                    cb_stat_sq,
                    ckl::ReservePolicy::None,
                    ckl::PushPolicy::None,
                    RECFG,
                    ckl::PackRelu::Disabled,
                    ckl::L1Accumulation::Disabled,
                    ckl::DestAccumulation::PerRow,
                    ckl::TileOffset::Strided)>{ckl::StridedTileRange{k, nc}});
        }
        if (has_tail) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(rows_t, 1),
                ckl::BinaryFpu<
                    in_spec,
                    mask_spec,
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::Row,
                    ckl::Dst::D0,
                    ckl::DestAccumulation::Disabled>{ckl::StridedTileRange{core_w - 1, CB_W_TILES}},
                ckl::Square<>{},
                ckl::PackTile<ckl::output(
                    cb_stat_sq,
                    ckl::ReservePolicy::None,
                    ckl::PushPolicy::None,
                    RECFG,
                    ckl::PackRelu::Disabled,
                    ckl::L1Accumulation::Disabled,
                    ckl::DestAccumulation::Disabled,
                    ckl::TileOffset::Strided)>{ckl::StridedTileRange{tail_col, nc}});
        }
        cb_push_back(cb_stat_sq, rows_t * nc);
    };

    auto phase_b = [&]() {
        const uint32_t c_full = core_w - has_tail;
        const uint32_t bulk_cols = (c_full + chunk_w - 1) / chunk_w;
        const uint32_t nc = bulk_cols + has_tail;
        if constexpr (VARIANT == V_ACCADD) {
            // The 1/N of reduce_mean is the CALLER's, so W_TRUE reproduces the op's
            // scaler-borne 1/W_true exactly (the AccumulateViaAdd datapath ignores
            // the scaler CB for an aligned reduce).
            ckl::reduce_mean<
                ckernel::ReduceDim::REDUCE_ROW,
                cb_stat_sq,
                cb_scaler,
                cb_out,
                ckl::ReduceInputPolicy::BulkWaitBulkPop,
                RMODE,
                ReduceFp32Mode::Fast,
                ckl::ReduceAlgorithm::AccumulateViaAdd>(
                ckl::ReduceInputBlockShape::of(rows_t, nc, 1), W_TRUE);
        } else {
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_stat_sq,
                cb_scaler,
                cb_out,
                ckl::ReduceInputPolicy::BulkWaitBulkPop,
                RMODE>(ckl::ReduceInputBlockShape::of(rows_t, nc, 1));
        }
    };

    // ---- THE CANDIDATE: one phase, accumulator never leaves DEST ----
    auto phase_merged = [&]() {
        const uint32_t c_full = core_w - has_tail;
        for (uint32_t r = 0; r < rows_t; ++r) {
            const uint32_t row_base = r * CB_W_TILES;
            tile_regs_acquire();
            // acc_to_dest from the FIRST op: a freshly acquired DEST reads 0, which is
            // exactly what eltwise_chain's DestAccumulation::PerRow relies on.
            if (!HOIST || has_tail) {
                mul_tiles_init(cb_in, cb_in, 1u, __builtin_LINE());
            }
            for (uint32_t g = 0; g < c_full; ++g) {
                const uint32_t t = row_base + g;
                mul_tiles(cb_in, cb_in, t, t, 0);
            }
            if (has_tail) {
                // RAGGED HIDDEN TILE: mask BEFORE squaring so a finite poison value in
                // the pad columns is annihilated instead of squared. The masked square
                // cannot go straight into the accumulator (the FPU has no
                // DEST+=DEST), so it lands in a transient DEST slot and is folded in
                // on the SFPU.
                reconfig_data_format_srcb(cb_in, cb_wmask);
                mul_bcast_rows_init_short(cb_in, cb_wmask);
                mul_tiles_bcast_rows(cb_in, cb_wmask, row_base + core_w - 1, 0, 1);
                reconfig_data_format_srcb(cb_wmask, cb_in);
                square_tile_init();
                square_tile(1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            // WITHIN-TILE COLUMN COLLAPSE, IN DEST. This is the pack->L1->unpack round
            // trip that the two-phase schedule pays; here it is one SFPU op on the
            // accumulator that is already sitting in DEST.
            if (!HOIST || has_tail) {
                ckernel::sfpu_reduce_init<ckernel::PoolType::SUM, dst_fmt>();
            }
            ckernel::sfpu_reduce<ckernel::PoolType::SUM, dst_fmt, ckernel::ReduceDim::REDUCE_ROW>(0, 1, 1);
            // 1/W_true, the divisor the op's reduce scaler carried. No
            // binop_with_scalar init needed straight after sfpu_reduce (same as
            // reduce_helpers_compute.inl's AVG finalize).
            if constexpr (CVALID_SCALE) {
                // Same op, same precision, HALF the SFPU vector ops: the RC vector
                // mode `mul_unary_tile` hardcodes is the one template argument the
                // compute API does not thread, so this is the API call with
                // VectorMode::C spelled out (the pattern rms_norm_compute.cpp already
                // uses for its column-valid Rsqrt / AddUnary finalize elements).
                MATH(SFPU_UNARY_CALL(
                    DST_SYNC_MODE,
                    DST_ACCUM_MODE,
                    calculate_binop_with_scalar,
                    (APPROX, ckernel::MUL_UNARY, 8 /* ITERATIONS */),
                    0,
                    VectorMode::C,
                    INV_W_BITS));
            } else if constexpr (!NO_SCALE) {
                mul_unary_tile(0, INV_W_BITS);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
    };

    if constexpr (HOIST) {
        mul_tiles_init(cb_in, cb_in, 1u, __builtin_LINE());
        ckernel::sfpu_reduce_init<ckernel::PoolType::SUM, dst_fmt>();
    }

    if constexpr (PHASE_ZONES) {
        for (uint32_t it = 0; it < ITERS; ++it) {
            if constexpr (MERGED) {
                MaybeDeviceZoneScope("cp_sumsq_merged");
                phase_merged();
            } else {
                {
                    MaybeDeviceZoneScope("cp_sumsq");
                    phase_a();
                }
                {
                    MaybeDeviceZoneScope("cp_reduce_stat");
                    phase_b();
                }
            }
            if (it + 1 < ITERS) {
                cb_wait_front(cb_out, rows_t);
                cb_pop_front(cb_out, rows_t);
            }
        }
    } else {
        MaybeDeviceZoneScope("cp_stats_all");
        for (uint32_t it = 0; it < ITERS; ++it) {
            if constexpr (MERGED) {
                phase_merged();
            } else {
                phase_a();
                phase_b();
            }
            if (it + 1 < ITERS) {
                cb_wait_front(cb_out, rows_t);
                cb_pop_front(cb_out, rows_t);
            }
        }
    }
}
"""


# =============================================================================
# Dataflow kernel — the two constant tiles the op's reader prepares for this stage:
# the reduce scaler (1/W_true, the non-standard scaler) and the 0/1 ragged-tile mask.
# =============================================================================
_CONST_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_scaler = 2;
    constexpr uint32_t cb_wmask = 3;
    constexpr uint32_t NEEDS_SCALER = get_compile_time_arg_val(0);
    constexpr uint32_t NEEDS_MASK = get_compile_time_arg_val(1);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(0);
    const uint32_t valid_elems = get_arg_val<uint32_t>(1);

    if constexpr (NEEDS_SCALER) {
        float inv_w;
        __builtin_memcpy(&inv_w, &inv_w_bits, sizeof(inv_w));
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            inv_w);
    }
    if constexpr (NEEDS_MASK) {
        dataflow_kernel_lib::prepare_reduce_mask<cb_wmask, ckernel::ReduceDim::REDUCE_ROW>(valid_elems);
    }
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def sharded_config(h_tiles, w_tiles):
    return ttnn.create_sharded_memory_config(
        shape=(h_tiles * TILE, w_tiles * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id, data_format, num_tiles):
    tile_size = ttnn.tile_size(data_format)
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=data_format, page_size=tile_size)
    return ttnn.CBDescriptor(total_size=tile_size * num_tiles, core_ranges=_single_core(), format_descriptors=[fmt])


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _rt(vals):
    rt = ttnn.RuntimeArgs()
    rt[0][0] = list(vals)
    return rt


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    variant,
    rows_t,
    core_w,
    valid_last=TILE,
    chunk_tiles=None,
    iters=1,
    math_fidelity=None,
    fp32_dest_acc_en=False,
):
    """One (variant, geometry) program over a resident rows_t x core_w bf16 block."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {tuple(VARIANTS)}, got {variant!r}")
    if not (1 <= valid_last <= TILE):
        raise ValueError("valid_last must be in [1, 32]")
    has_tail = 1 if valid_last != TILE else 0
    chunk = core_w if chunk_tiles is None else chunk_tiles
    w_true = (core_w - has_tail) * TILE + (valid_last if has_tail else 0)

    c_full = core_w - has_tail
    bulk_cols = (c_full + chunk - 1) // chunk
    nc = bulk_cols + has_tail

    merged = variant in _MERGED
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[VARIANTS[variant], core_w, chunk, iters, _f32_bits(1.0 / w_true), w_true],
        runtime_args=_rt([rows_t, core_w, has_tail]),
        # The USER's precision contract, identical for every variant.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=math_fidelity or ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_dest_acc_en
        ),
    )
    const = ttnn.KernelDescriptor(
        kernel_source=_CONST_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        # FAIRNESS: the scaler tile is prepared for EVERY variant, including the
        # merged ones that never read it. The op's reader fills it once per kernel,
        # off this stage's critical path; charging it only to the baseline would be a
        # measurement artifact on NCRISC, not a property of the two schedules. The L1
        # the merged form gives back (cb_scaler + cb_stat_sq) is reported separately.
        compile_time_args=[1, has_tail],
        runtime_args=_rt([_f32_bits(1.0 / w_true), valid_last if has_tail else TILE]),
        config=ttnn.ReaderConfigDescriptor(),
    )

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
    ]
    cbs.append(_scratch_cb(CB_SCALER, ttnn.bfloat16, 1))  # see the FAIRNESS note above
    if not merged:
        # The L1 the candidate gives back (reported, not measured here).
        cbs.append(_scratch_cb(CB_STAT_SQ, ttnn.bfloat16, rows_t * nc))
    if has_tail:
        cbs.append(_scratch_cb(CB_WMASK, ttnn.bfloat16, 1))

    return ttnn.ProgramDescriptor(kernels=[const, compute], semaphores=[], cbs=cbs)


def run_variant(input_tensor, *, variant, rows_t, core_w, valid_last=TILE, chunk_tiles=None, iters=1, **kw):
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows_t * TILE, TILE]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        sharded_config(rows_t, 1),
    )
    descriptor = create_program_descriptor(
        input_tensor,
        output,
        variant=variant,
        rows_t=rows_t,
        core_w=core_w,
        valid_last=valid_last,
        chunk_tiles=chunk_tiles,
        iters=iters,
        **kw,
    )
    return ttnn.generic_op([input_tensor, output], descriptor)

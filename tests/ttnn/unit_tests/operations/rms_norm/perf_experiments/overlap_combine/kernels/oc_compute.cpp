// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// overlap_combine bench — compute.  THE VARIANT LIVES HERE.
//
// Three stages, exactly as the op has them on a hidden-split sharded plan:
//   stat_pass(sb)     Sum(x^2) over STAT_ROWS tile-rows  -> STAT_ROWS fp32 tiles
//   combine_pass(sb)  [root only] reduce over (STAT_ROWS, s) gathered + finalize
//   apply_pass(sb)    x *= 1/rms  (BroadcastDim::Col) -> the resident output shard
//
// Two knobs select the schedule under test:
//
//   PIPELINE = 0  the op's CURRENT structure — strictly serial per block:
//                 stat(b) -> [combine(b)] -> wait 1/rms(b) -> apply(b).
//                 Nothing on a core overlaps the combine round trip.
//   PIPELINE = 1  software-pipelined: stat(0) in a prologue, then per block
//                 stat(b+1) BEFORE waiting for 1/rms(b).  Needs depth-2
//                 cb_sq_partials / cb_gathered_partials / cb_rms_bcast /
//                 cb_rms_recip (the host sizes them).
//
//   STAT_ROWS > APPLY_ROWS  decouples the STAT block from the APPLY block: one
//                 coarse combine round trip covers STAT_ROWS tile-rows while the
//                 apply pass still walks APPLY_ROWS at a time (the apply pass is
//                 L1/DEST-bound, the combine is latency-bound, so they do not
//                 want the same granularity).
//
// -----------------------------------------------------------------------------
// WHY EVERY READ CARRIES AN ABSOLUTE TILE BASE (`TileOffset::Set`)
// -----------------------------------------------------------------------------
// The op walks the resident shard with the CB's READ POINTER: `x_held` is
// (WaitPolicy::None, PopPolicy::None, OperandKind::Block, TileOffset::Unset), so
// its tile index is the walk index measured from wherever the read pointer sits,
// and compute advances it with one `cb_pop_front(BLOCK_TILES)` per block (the
// reader re-pushes to keep the CB full).
//
// That representation cannot express the pipelined order AT ALL: `stat(b+1)` has
// to read block b+1 while block b's window is still live for `apply(b)`, and a
// single read pointer cannot be in two places.  So both passes here take an
// explicit base (`sb * STAT_TILES`, `sb * STAT_TILES + j * APPLY_TILES`) and the
// shard CB is waited ONCE and never popped.  On a resident shard this is a pure
// no-op transformation of the baseline (the pop/re-push pair moves no data), and
// it is applied IDENTICALLY to the baseline variant so the measured delta is the
// schedule alone.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_rms_bcast = 5;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_scaler = 7;
constexpr uint32_t cb_output_tiles = 9;

// Largest d <= cap that divides `width` (same rule the op uses: the grid walk
// blocks row-wise, so a DEST group wider than the row leaves a short tail).
constexpr uint32_t dest_block_divisor(uint32_t width, uint32_t cap) {
    for (uint32_t d = (cap < width ? cap : width); d > 1; --d) {
        if (width % d == 0) {
            return d;
        }
    }
    return 1;
}

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);           // slice hidden tiles
    constexpr uint32_t STAT_ROWS = get_compile_time_arg_val(1);   // SB
    constexpr uint32_t APPLY_ROWS = get_compile_time_arg_val(2);  // B
    constexpr uint32_t NUM_SLICES = get_compile_time_arg_val(3);  // s
    constexpr uint32_t SHARD_TILES = get_compile_time_arg_val(4);
    constexpr uint32_t DEST_BLOCK_TILES = get_compile_time_arg_val(5);
    constexpr uint32_t PIPELINE = get_compile_time_arg_val(6);

    const uint32_t num_stat_blocks = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(2);
    const uint32_t eps_bits = get_arg_val<uint32_t>(3);

    constexpr uint32_t STAT_TILES = STAT_ROWS * S;
    constexpr uint32_t APPLY_TILES = APPLY_ROWS * S;
    constexpr uint32_t SUBS = STAT_ROWS / APPLY_ROWS;  // apply sub-blocks per stat block
    static_assert(STAT_ROWS % APPLY_ROWS == 0, "the apply block must tile the stat block");

    constexpr uint32_t DEST_BLOCK = dest_block_divisor(S, DEST_BLOCK_TILES);

    // Same crossover dispatch the op uses for the root's combine datapath.
    constexpr uint32_t COMBINE_ACCUMULATE_MIN_TILES = 4;
    constexpr auto COMBINE_ALGORITHM = NUM_SLICES >= COMBINE_ACCUMULATE_MIN_TILES
                                           ? ckl::ReduceAlgorithm::AccumulateViaAdd
                                           : ckl::ReduceAlgorithm::Auto;

    // ---- operand configurations ----
    // x, addressed ABSOLUTELY inside the resident shard (see the header note).
    constexpr auto x_at = ckl::input(
        cb_input_tiles, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block, ckl::TileOffset::Set);
    // 1/rms: a REDUCE_ROW result is column-shaped, so it broadcasts back across
    // columns (Col).  Caller-managed wait/pop because the coarse-stat variant
    // consumes the SB-tile window in SUBS sub-blocks, each at its own base.
    constexpr auto rms_col_at = ckl::input(
        cb_rms_recip,
        ckl::BroadcastDim::Col,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::OperandKind::Col,
        ckl::TileOffset::Set);
    // Sum(x^2)'s output: one fp32 tile per tile-row, folded in DEST (no x^2 tile
    // is ever materialized).  Mirrors `ckl::sum_of_squares`' own output spec.
    constexpr auto sq_out = ckl::output(
        cb_sq_partials,
        ckl::ReservePolicy::PerOuter,
        ckl::PushPolicy::PerOuter,
        ckl::DataFormatReconfig::Enabled,
        ckl::PackRelu::Disabled,
        ckl::L1Accumulation::Disabled,
        ckl::DestAccumulation::PerRow);
    // The resident output shard.  Batched lifecycle to match the DEST window.
    constexpr auto to_output =
        ckl::output(cb_output_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);

    compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles);

    // The resident shard: waited ONCE, never popped.  Zero NoC reads behind it.
    {
        MaybeDeviceZoneScope("cp_wait_in");
        cb_wait_front(cb_input_tiles, SHARD_TILES);
    }

    // mean = Sum(x^2) * (1/W) using the TRUE element count, then + eps, then rsqrt.
    auto finalize = [inv_w_bits, eps_bits](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, inv_w_bits);
        add_unary_tile(dst_idx, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    auto stat_pass = [](uint32_t sb) {
        MaybeDeviceZoneScope("cp_sumsq");
        const uint32_t base = sb * STAT_TILES;
        ckl::eltwise_chain(
            ckl::IterationShape::grid(STAT_ROWS, S),
            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_at, x_at, ckl::Dst::D0, ckl::DestAccumulation::PerRow>{base, base},
            ckl::PackTile<sq_out>{});
    };

    auto combine_pass = [&finalize]() {
        {
            // Hoisted out of the reduce so `cp_combine` is the reduce PAYLOAD alone
            // and this zone is the gather incast the pipeline is trying to hide.
            MaybeDeviceZoneScope("cp_combine_wait");
            cb_wait_front(cb_gathered_partials, NUM_SLICES * STAT_ROWS);
        }
        MaybeDeviceZoneScope("cp_combine");
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_gathered_partials,
            cb_scaler,
            cb_rms_bcast,
            ckl::ReduceInputPolicy::BulkWaitBulkPop,
            ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Fast,
            COMBINE_ALGORITHM,
            ckl::NoAccumulation,
            decltype(finalize)>(
            ckl::ReduceInputBlockShape::of(STAT_ROWS, NUM_SLICES),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            finalize);
    };

    auto apply_pass = [](uint32_t sb) {
        {
            // On a serial loop this is where a non-root core's WHOLE wall sits.
            MaybeDeviceZoneScope("cp_rms_wait");
            cb_wait_front(cb_rms_recip, STAT_ROWS);
        }
        {
            MaybeDeviceZoneScope("cp_scale");
            for (uint32_t j = 0; j < SUBS; ++j) {
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(APPLY_ROWS, S).block_size(DEST_BLOCK),
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_at, rms_col_at>{
                        sb * STAT_TILES + j * APPLY_TILES, j * APPLY_ROWS},
                    ckl::PackTile<to_output>{});
            }
        }
        cb_pop_front(cb_rms_recip, STAT_ROWS);
    };

    if constexpr (PIPELINE) {
        // Prologue: block 0's stat leaves before anyone waits for anything.
        stat_pass(0);
        for (uint32_t sb = 0; sb < num_stat_blocks; ++sb) {
            // The whole idea: block sb+1's Sum(x^2) has NO unmet dependency (x is
            // resident), so it runs — and its stat reaches the root — while block
            // sb's combine round trip is still in flight.
            if (sb + 1 < num_stat_blocks) {
                stat_pass(sb + 1);
            }
            if (is_root) {
                combine_pass();
            }
            apply_pass(sb);
        }
    } else {
        for (uint32_t sb = 0; sb < num_stat_blocks; ++sb) {
            stat_pass(sb);
            if (is_root) {
                combine_pass();
            }
            apply_pass(sb);
        }
    }
}

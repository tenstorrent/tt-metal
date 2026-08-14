// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH — rms_norm's cross-core combine, compute half.
//
// Reconstructs exactly two compute phases of the real op and nothing else:
//   * `square_accumulate_block`  -> ckl::sum_of_squares over (B, S)   [EVERY core]
//   * `combine_block`            -> ckl::reduce<SUM, REDUCE_ROW> + finalize
//                                  over the gathered (rows, s) block  [the REDUCER core(s)]
//
// The only difference between the baseline and the candidate is HOW MANY ROWS a
// core reduces (`OWN_ROWS`) and how many cores are reducers:
//   baseline (flat root):      OWN_ROWS = B,           1 reducer  per row-group
//   candidate (reduce-scatter): OWN_ROWS = B/num_owners, num_owners reducers
// Same helper, same template params, same precision contract.
//
// The trailing `cb_wait_front(cb_rms_recip)` / pop is NOT decoration: in the real
// op the broadcast-back is what gates a contributor's NEXT block (its scale phase
// waits on 1/rms before its next Sum(x^2) can push partials), and that gate is
// what keeps a fast contributor from overwriting a reducer's gather pages one
// block early.  The bench has no scale phase, so the gate is restored here
// explicitly — identically for every variant.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_gathered = 4;
constexpr uint32_t cb_stat_out = 5;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_scaler = 7;

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);         // hidden tiles per core
    constexpr uint32_t B = get_compile_time_arg_val(1);         // rows per block
    constexpr uint32_t NSLICE = get_compile_time_arg_val(2);    // s — cores per row-group
    constexpr uint32_t OWN_ROWS = get_compile_time_arg_val(3);  // rows THIS core reduces
    constexpr uint32_t IN_WAIT_TILES = get_compile_time_arg_val(4);
    constexpr uint32_t BLOCK_TILES = B * S;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t is_owner = get_arg_val<uint32_t>(1);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(2);
    const uint32_t eps_bits = get_arg_val<uint32_t>(3);

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_stat_out);

    // Same fan-in dispatch as the real op: pairwise DEST accumulate above the
    // crossover, the matmul-with-ones reduce below it.  Depends on `s` only, so it
    // is identical for the baseline and the candidate at a given geometry.
    constexpr uint32_t COMBINE_ACCUMULATE_MIN_TILES = 4;
    constexpr auto COMBINE_ALGORITHM =
        NSLICE >= COMBINE_ACCUMULATE_MIN_TILES ? ckl::ReduceAlgorithm::AccumulateViaAdd : ckl::ReduceAlgorithm::Auto;

    constexpr auto x_held = ckl::input(cb_in, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block);
    constexpr auto block_shape = ckl::IterationShape::grid(B, S);

    auto finalize = [inv_w_bits, eps_bits](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, inv_w_bits);
        add_unary_tile(dst_idx, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    for (uint32_t block = 0; block < num_blocks; ++block) {
        {
            MaybeDeviceZoneScope("cp_wait_in");
            cb_wait_front(cb_in, IN_WAIT_TILES);
        }
        {
            MaybeDeviceZoneScope("cp_sumsq");
            ckl::sum_of_squares<x_held, ckl::row_output(cb_sq_partials)>(block_shape);
        }

        if (is_owner) {
            {
                // Hoisted out of the reduce helper's internal BulkWait so `cp_combine`
                // measures the reduce PAYLOAD and this zone measures the gather incast.
                MaybeDeviceZoneScope("cp_combine_wait");
                cb_wait_front(cb_gathered, NSLICE * OWN_ROWS);
            }
            MaybeDeviceZoneScope("cp_combine");
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_gathered,
                cb_scaler,
                cb_stat_out,
                ckl::ReduceInputPolicy::BulkWaitBulkPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                COMBINE_ALGORITHM,
                ckl::NoAccumulation,
                decltype(finalize)>(
                ckl::ReduceInputBlockShape::of(OWN_ROWS, NSLICE),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::NoAccumulation{},
                finalize);
        }

        // The real op's `cp_rms_wait` — the broadcast gate (see the header note).
        {
            MaybeDeviceZoneScope("cp_rms_wait");
            cb_wait_front(cb_rms_recip, B);
        }
        cb_pop_front(cb_rms_recip, B);
        cb_pop_front(cb_in, BLOCK_TILES);
    }
}

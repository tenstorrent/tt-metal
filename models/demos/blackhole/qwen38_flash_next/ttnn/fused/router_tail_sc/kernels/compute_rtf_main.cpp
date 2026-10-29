// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Router tail, fast sort-core form (QWEN38_ROUTER_TAIL_FAST=1), compute of core A (compute_rt_main.cpp + RTF_NOSTART)
// (rows == 1).  Today's softmax, sum and division, instruction for instruction
// (../../router_tail/kernels/compute_router_tail.cpp); the sort lives on core B (compute_rt_sort.cpp).  Core A's reader
// gathers the ten probabilities of the indices B sends and builds the [token, k] tiles cb_vals (zero padded) and
// cb_pad_div, so no transpose / pack of the sorted values happens here. Runtime arg 0 (pass_mask) is unused, arg 1
// (token_mask) selects the live exp vector pairs, as in today's kernel.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "../../router_tail/kernels/exp_live.h"
#include "../../kernels/zones.h"
#include "../../kernels/source_reciprocal.h"

#ifndef RTF_NOTOKEN
#define RTF_NOTOKEN 0  // the reader pushes cb_vals complete: no zero-fill token
#endif
#ifndef RTF_NOSTART
#define RTF_NOSTART 0
#endif
#ifndef FRT_EXP_LIVE
#define FRT_EXP_LIVE 1
#endif

void kernel_main() {
    const uint32_t token_mask = get_arg_val<uint32_t>(1);
    const uint32_t live_pairs = exp_live_pairs(token_mask);
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_max_scaler = get_named_compile_time_arg_val("cb_max_scaler");
    constexpr uint32_t cb_sum_scaler = get_named_compile_time_arg_val("cb_sum_scaler");
    constexpr uint32_t cb_norm_scaler = get_named_compile_time_arg_val("cb_norm_scaler");
    constexpr uint32_t cb_max = get_named_compile_time_arg_val("cb_max");
    constexpr uint32_t cb_exps = get_named_compile_time_arg_val("cb_exps");
    constexpr uint32_t cb_recip = get_named_compile_time_arg_val("cb_recip");
    constexpr uint32_t cb_probs = get_named_compile_time_arg_val("cb_probs");
    constexpr uint32_t cb_vals = get_named_compile_time_arg_val("cb_vals");
    constexpr uint32_t cb_vals_ready = get_named_compile_time_arg_val("cb_vals_ready");
    constexpr uint32_t cb_pad_div = get_named_compile_time_arg_val("cb_pad_div");
    constexpr uint32_t cb_sums = get_named_compile_time_arg_val("cb_sums");
    constexpr uint32_t cb_sums_ready = get_named_compile_time_arg_val("cb_sums_ready");
    constexpr uint32_t cb_scores = get_named_compile_time_arg_val("cb_scores");
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr uint32_t ndst = 4;  // softmax block size under fp32 dest
    static_assert(Wt % ndst == 0, "the softmax blocks must tile the row");

    DataflowBuffer in0(cb_in0);
    DataflowBuffer max_scaler(cb_max_scaler);
    DataflowBuffer sum_scaler(cb_sum_scaler);
    DataflowBuffer maxv(cb_max);
    DataflowBuffer exps(cb_exps);
    DataflowBuffer recips(cb_recip);
    DataflowBuffer probs(cb_probs);
    DataflowBuffer pad_div(cb_pad_div);

    {
        FUSED_ZONE("fz_rt_c_softmax");
        // ---- softmax.cpp: NUMERIC_STABLE, no mask, EXP_APPROX 0 ----
        compute_kernel_hw_startup(cb_in0, cb_max_scaler, cb_exps);
        max_scaler.wait_front(1);
        sum_scaler.wait_front(1);
        reconfig_data_format(cb_in0, cb_in0);
        pack_reconfig_data_format(cb_exps);
        copy_init(cb_in0);

        compute_kernel_lib::reduce<
            PoolType::MAX,
            ReduceDim::REDUCE_ROW,
            cb_in0,
            cb_max_scaler,
            cb_max,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
            compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        exp_tile_init<false>();
        reconfig_data_format_srcb(cb_max);
        maxv.wait_front(1);
        sub_bcast_cols_init(cb_in0, cb_max);
        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            tile_regs_acquire();
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                sub_tiles_bcast_cols(cb_in0, cb_max, wt + wt8, 0, wt8);
            }
            exps.reserve_back(ndst);
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
#if FRT_EXP_LIVE
                exp_tile_live(wt8, live_pairs);  // the full precise exp's instructions, on the live vector pairs only
#else
                exp_tile<false>(wt8);
#endif
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                pack_tile(wt8, cb_exps);
            }
            tile_regs_release();
            exps.push_back(ndst);
        }
        in0.pop_front(Wt);
        maxv.pop_front(1);
        exps.wait_front(Wt);

        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_exps,
            cb_sum_scaler,
            cb_recip,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            compute_kernel_lib::ReduceInputBlockShape::row(Wt),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t) {
                qwen38_recip_tile_init();
                qwen38_recip_tile(0);
            });

        recips.wait_front(1);
        reconfig_data_format(cb_exps, cb_recip);
        pack_reconfig_data_format(cb_probs);
        mul_bcast_cols_init(cb_exps, cb_recip);
        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            tile_regs_acquire();
            probs.reserve_back(ndst);
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                mul_tiles_bcast<BroadcastType::COL>(cb_exps, cb_recip, wt + wt8, 0, wt8);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                pack_tile(wt8, cb_probs);
            }
            tile_regs_release();
            probs.push_back(ndst);
        }
        recips.pop_front(1);
        exps.pop_front(Wt);
        max_scaler.pop_front(1);
        sum_scaler.pop_front(1);
    }

    {
        FUSED_ZONE("fz_rt_c_sum");
        // ---- reduce.cpp: SUM REDUCE_ROW, Ht 1, Wt 1, NC 1, enable_fp32_sfpu 1 (ttnn.sum on fp32 with fp32 dest) ----
        DataflowBuffer norm_scaler(cb_norm_scaler);
        DataflowBuffer vals_ready(cb_vals_ready);
        compute_kernel_hw_startup(cb_vals, cb_norm_scaler, cb_sums);
#if !RTF_NOTOKEN
        vals_ready.wait_front(1);
#endif
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_vals,
            cb_norm_scaler,
            cb_sums,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
            ReduceFp32Mode::Accurate>(
            compute_kernel_lib::ReduceInputBlockShape::of(1, 1, 1),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            compute_kernel_lib::NoOp{});
        norm_scaler.pop_front(1);
#if !RTF_NOTOKEN
        vals_ready.pop_front(1);
#endif
    }

    // ---- eltwise_binary_sfpu.cpp DIV (lhs DST 0, rhs DST 1: the sums tile with column 0 broadcast in place by the
    // reader), then the typecast op's fp32 -> bf16 ----
    {
        FUSED_ZONE("fz_rt_c_div");
        DataflowBuffer sums(cb_sums);
        DataflowBuffer sums_ready(cb_sums_ready);
        DataflowBuffer scores(cb_scores);
#if !RTF_NOSTART
        compute_kernel_hw_startup(cb_pad_div, cb_scores);
#endif
        copy_init(cb_pad_div);
        div_binary_tile_init();
        pad_div.wait_front(1);
        sums_ready.wait_front(1);
        scores.reserve_back(1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_sums, cb_pad_div);
        copy_init(cb_pad_div);
        copy_tile(cb_pad_div, 0, 0);
        reconfig_data_format_srca(cb_pad_div, cb_sums);
        copy_init(cb_sums);
        copy_tile(cb_sums, 0, 1);
        div_binary_tile(0, 1, 0);
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_scores);
        pack_tile(0, cb_scores);
        tile_regs_release();
        scores.push_back(1);
        pad_div.pop_front(1);
        sums.pop_front(1);
        sums_ready.pop_front(1);
    }
}

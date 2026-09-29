// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Router tail, compute: one 32-row tile of fp32 logits -> the top-10 renormalized bf16 scores ([token, k] tile) and
// their uint32 indices ([k, token] tile).  Each phase issues the instruction sequence of the ttnn op it replaces,
// on CBs of the same data format and unpack mode, so the arithmetic (and the top-k tie order) is the chain's:
//   softmax.cpp (numeric stable, no mask, fp32 dest, exp/recip precise)      -> cb_probs
//   topk.cpp single-core insertion sort, Wt=16, one output tile, unstable   -> cb_vals_t / cb_idx_t (transposed)
//   transpose_and_pack of the values                                        -> cb_vals + cb_pad_div ([token, k])
//   reduce.cpp SUM REDUCE_ROW, accurate fp32 SFPU path, on cb_vals once the reader zeroed its padding -> cb_sums
//   eltwise_binary_sfpu.cpp DIV (in0 * recip(in1)) then typecast fp32->bf16 -> cb_scores
// The index tiles arrive pre-transposed (row k of tile w = w*32+k) and are copied into DST: the same DST content
// the kernel's transpose produces, without the transpose.
//
// Runtime arg 0, pass_mask: 0 runs the LLK's four-pass local sort (the single-core form); bits 0..3 run only those
// passes of the same network (topk_lanes.h), each pass being the complete sort of eight token columns.  The lane
// form gives every core one pass and the eight tokens it holds; the other columns are left unsorted and never read.
// Runtime arg 1, token_mask: the token rows this core produces; the precise exp runs over the SFPU vector pairs that
// hold them only (exp_live.h; the other rows' exps are never read), unless FRT_EXP_LIVE is 0 (the full exp_tile, A/B).

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "topk_lanes.h"
#include "../../kernels/zones.h"
#include "exp_live.h"

// One tile of the insertion chain into DST: the probabilities tile transposed into value slot `slot`, its
// pre-transposed index tile copied into slot + 2 (topk.cpp loads both the same way).
FORCE_INLINE void frt_load_tile(uint32_t cb_probs, uint32_t cb_index, uint32_t w, uint32_t slot) {
    reconfig_data_format_srca(cb_probs);
    transpose_init(cb_probs);
    transpose_tile(cb_probs, w, slot);
    reconfig_data_format_srca(cb_index);
    copy_init(cb_index);
    copy_tile(cb_index, w, slot + 2);
}

// Dev knobs (timing only; every one of them breaks the output): FRT_SORT_PHASE_START..FRT_SORT_PHASE_END runs that
// window of the network's phases (0..5; a single phase runs all its steps), FRT_EXP_ITERATIONS runs N of the 8 SFPU
// vectors per face of the precise exp (0 skips the exp).
#ifndef FRT_SORT_PHASE_START
#define FRT_SORT_PHASE_START 0
#endif
#ifndef FRT_SORT_PHASE_END
#define FRT_SORT_PHASE_END 5
#endif
#ifndef FRT_EXP_ITERATIONS
#define FRT_EXP_ITERATIONS 8
#endif
#ifndef FRT_EXP_LIVE
#define FRT_EXP_LIVE 1
#endif

// The chain's sort of the 64 values in DST 0/1 (indices 2/3): topk.cpp's local sort call (unstable network, largest,
// end phase 5); pass_mask != 0 sorts only those token passes of the same network (topk_lanes.h).
FORCE_INLINE void frt_sort64(uint32_t pass_mask) {
#ifndef FRT_TOPK_SORT_SKIP  // dev knob (timing): transposes and copies only, no sort
#if FRT_SORT_PHASE_START == 0 && FRT_SORT_PHASE_END == 5
    if (pass_mask == 0) {
        ckernel::topk_local_sort<false>(0, 0 /* largest */, 5 /* end_phase */);
    } else {
        topk_local_sort_lanes<false>(0, 0 /* largest */, 5 /* end_phase */, pass_mask);
    }
#else
    // the phase window: a single phase (start == end) runs its steps num_steps..4 down to 1 as the full network does
    if (pass_mask == 0) {
        ckernel::topk_local_sort<false>(
            0, 0 /* largest */, FRT_SORT_PHASE_END, FRT_SORT_PHASE_START, 4, FRT_SORT_PHASE_END + 1);
    } else {
        topk_local_sort_lanes<false>(
            0, 0 /* largest */, FRT_SORT_PHASE_END, pass_mask, FRT_SORT_PHASE_START, 4, FRT_SORT_PHASE_END + 1);
    }
#endif
#endif
}

void kernel_main() {
    const uint32_t pass_mask = get_arg_val<uint32_t>(0);
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
    constexpr uint32_t cb_index = get_named_compile_time_arg_val("cb_index");
    constexpr uint32_t cb_vals_t = get_named_compile_time_arg_val("cb_vals_t");
    constexpr uint32_t cb_idx_t = get_named_compile_time_arg_val("cb_idx_t");
    constexpr uint32_t cb_vals = get_named_compile_time_arg_val("cb_vals");
    constexpr uint32_t cb_vals_ready = get_named_compile_time_arg_val("cb_vals_ready");
    constexpr uint32_t cb_pad_div = get_named_compile_time_arg_val("cb_pad_div");
    constexpr uint32_t cb_sums = get_named_compile_time_arg_val("cb_sums");
    constexpr uint32_t cb_sums_ready = get_named_compile_time_arg_val("cb_sums_ready");
    constexpr uint32_t cb_scores = get_named_compile_time_arg_val("cb_scores");
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr uint32_t ndst = 4;  // softmax block size under fp32 dest
    static_assert(Wt % ndst == 0, "the softmax blocks must tile the row");
#ifndef FRT_TOPK_TILES
#define FRT_TOPK_TILES Wt
#endif
    constexpr uint32_t topk_tiles = FRT_TOPK_TILES;  // dev knob: sort only the first N width tiles (timing)

    DataflowBuffer in0(cb_in0);
    DataflowBuffer max_scaler(cb_max_scaler);
    DataflowBuffer sum_scaler(cb_sum_scaler);
    DataflowBuffer maxv(cb_max);
    DataflowBuffer exps(cb_exps);
    DataflowBuffer recips(cb_recip);
    DataflowBuffer probs(cb_probs);

    {
        FUSED_ZONE("fz_rt_c_softmax");
        // ---- softmax.cpp: NUMERIC_STABLE, no mask, EXP_APPROX 0 ----
        compute_kernel_hw_startup(cb_in0, cb_max_scaler, cb_exps);
#ifdef FRT_SOFTMAX_COPY_ONLY
        // dev knob (timing): probabilities = logits, no softmax math
        max_scaler.wait_front(1);
        sum_scaler.wait_front(1);
        in0.wait_front(Wt);
        copy_init(cb_in0);
        pack_reconfig_data_format(cb_probs);
        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            tile_regs_acquire();
            probs.reserve_back(ndst);
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                copy_tile(cb_in0, wt + wt8, wt8);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                pack_tile(wt8, cb_probs);
            }
            tile_regs_release();
            probs.push_back(ndst);
        }
        in0.pop_front(Wt);
        max_scaler.pop_front(1);
        sum_scaler.pop_front(1);
#else
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
#if FRT_EXP_ITERATIONS == 8 && FRT_EXP_LIVE
                exp_tile_live(wt8, live_pairs);  // the full precise exp's instructions, on the live vector pairs only
#elif FRT_EXP_ITERATIONS == 8
                exp_tile<false>(wt8);
#elif FRT_EXP_ITERATIONS > 0
                exp_tile<false, false, ckernel::InputClamping::ClampToNegative, FRT_EXP_ITERATIONS>(wt8);  // dev knob
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
                recip_tile_init();
                recip_tile(0);
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
#endif
    }

    // ---- topk.cpp single core, Wt tiles, output_tiles 1, largest, stable_sort false ----
    // The running top 32 stays in DST 0 (values) / DST 2 (indices) instead of the kernel's result_prep CB round
    // trip; every new tile transposes into DST 1 / DST 3 exactly as the kernel's insertion loads it.
    DataflowBuffer index(cb_index);
    DataflowBuffer vals_t(cb_vals_t);
    DataflowBuffer idx_t(cb_idx_t);
    compute_kernel_hw_startup(cb_probs, cb_index, cb_vals_t);
    ckernel::topk_tile_init();
    probs.wait_front(Wt);
    index.wait_front(Wt);
    {
        FUSED_ZONE("fz_rt_c_topk_sort");
#ifndef FRT_TOPK_SPLIT
        tile_regs_acquire();
        for (uint32_t w = 0; w < topk_tiles; ++w) {
            frt_load_tile(cb_probs, cb_index, w, (w == 0) ? 0 : 1);
            if (w != 0) {
                frt_sort64(pass_mask);
            }
        }
        tile_regs_commit();
#else
        // dev knob (study, never serves): a two-core width split emulated on one core.  The chain over tiles 0..Wt/2-1
        // (its running top 32 staged through cb_vals_t / cb_idx_t), the chain over tiles Wt/2..Wt-1, then one sort of
        // the two running sets.  The values agree with the sequential chain, the tie order does not.
        static_assert(FRT_TOPK_SPLIT == 2, "the split emulation is two-way");
        constexpr uint32_t half = Wt / 2;
        tile_regs_acquire();
        for (uint32_t w = 0; w < half; ++w) {
            frt_load_tile(cb_probs, cb_index, w, (w == 0) ? 0 : 1);
            if (w != 0) {
                frt_sort64(pass_mask);
            }
        }
        tile_regs_commit();
        vals_t.reserve_back(1);
        idx_t.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_vals_t);
        pack_tile(0, cb_vals_t);
        pack_reconfig_data_format(cb_idx_t);
        pack_tile(2, cb_idx_t);
        tile_regs_release();
        vals_t.push_back(1);
        idx_t.push_back(1);
        tile_regs_acquire();
        for (uint32_t w = half; w < Wt; ++w) {
            frt_load_tile(cb_probs, cb_index, w, (w == half) ? 0 : 1);
            if (w != half) {
                frt_sort64(pass_mask);
            }
        }
        vals_t.wait_front(1);
        idx_t.wait_front(1);
        reconfig_data_format_srca(cb_vals_t);
        copy_init(cb_vals_t);
        copy_tile(cb_vals_t, 0, 1);
        reconfig_data_format_srca(cb_idx_t);
        copy_init(cb_idx_t);
        copy_tile(cb_idx_t, 0, 3);
        frt_sort64(pass_mask);
        tile_regs_commit();
        vals_t.pop_front(1);
        idx_t.pop_front(1);
#endif
    }
    probs.pop_front(Wt);
    index.pop_front(Wt);
    vals_t.reserve_back(1);
    idx_t.reserve_back(1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_vals_t);
    pack_tile(0, cb_vals_t);
    pack_reconfig_data_format(cb_idx_t);
    pack_tile(2, cb_idx_t);
    tile_regs_release();
    vals_t.push_back(1);
    idx_t.push_back(1);

    // transpose_and_pack(result_prep_val -> output_val): the [token, k] tile, once for the sum (the reader zeroes
    // its padding in place) and once for the division
    DataflowBuffer vals(cb_vals);
    DataflowBuffer pad_div(cb_pad_div);
    vals_t.wait_front(1);
    reconfig_data_format_srca(cb_vals_t);
    transpose_init(cb_vals_t);
    pack_reconfig_data_format(cb_vals);
    tile_regs_acquire();
    transpose_tile(cb_vals_t, 0, 0);
    tile_regs_commit();
    vals.reserve_back(1);
    pad_div.reserve_back(1);
    tile_regs_wait();
    pack_tile(0, cb_vals);
    pack_tile(0, cb_pad_div);
    tile_regs_release();
    vals.push_back(1);
    pad_div.push_back(1);
    vals_t.pop_front(1);

    {
        FUSED_ZONE("fz_rt_c_sum");
        // ---- reduce.cpp: SUM REDUCE_ROW, Ht 1, Wt 1, NC 1, enable_fp32_sfpu 1 (ttnn.sum on fp32 with fp32 dest) ----
        DataflowBuffer norm_scaler(cb_norm_scaler);
        DataflowBuffer vals_ready(cb_vals_ready);
        compute_kernel_hw_startup(cb_vals, cb_norm_scaler, cb_sums);
        vals_ready.wait_front(1);
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
        vals_ready.pop_front(1);
    }

    // ---- eltwise_binary_sfpu.cpp DIV (lhs DST 0, rhs DST 1: the sums tile with column 0 broadcast in place by the
    // reader), then the typecast op's fp32 -> bf16 ----
    {
        FUSED_ZONE("fz_rt_c_div");
        DataflowBuffer sums(cb_sums);
        DataflowBuffer sums_ready(cb_sums_ready);
        DataflowBuffer scores(cb_scores);
        compute_kernel_hw_startup(cb_pad_div, cb_scores);
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

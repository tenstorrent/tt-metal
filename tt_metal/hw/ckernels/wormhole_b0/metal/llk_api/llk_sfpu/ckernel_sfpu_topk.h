// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_topk.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_sfpu_op.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_phases_steps(
    uint idir, uint i_end_phase, uint i_start_phase, uint i_end_step, uint i_start_step) {
    _bitonic_topk_phases_steps<APPROXIMATION_MODE, is_fp32_dest_acc_en, STABLE_SORT>(
        idir, i_end_phase, i_start_phase, i_end_step, i_start_step);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool idir = false, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_merge(uint m_iter, uint k) {
    _bitonic_topk_merge<APPROXIMATION_MODE, is_fp32_dest_acc_en, idir, STABLE_SORT>(m_iter, k);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_rebuild(uint idir, uint m_iter, uint k, uint logk, uint skip_second) {
    _bitonic_topk_rebuild<APPROXIMATION_MODE, is_fp32_dest_acc_en, STABLE_SORT>(idir, m_iter, k, logk, skip_second);
}

template <bool APPROXIMATION_MODE>
inline void topk_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 32}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    _init_topk();
}

// TopkLocalSort / TopkMerge / TopkRebuild <APPROX, ..., DST_SYNC, DST_ACCUM>: topk_local_sort, topk_merge,
// topk_rebuild and topk_tile_init (compute_kernel_api.h). All three stages share topk_init.
template <bool APPROXIMATION_MODE, bool STABLE_SORT, DstSync DST_SYNC, bool DST_ACCUM>
struct TopkLocalSort
    : SfpuUnaryOp<TopkLocalSort<APPROXIMATION_MODE, STABLE_SORT, DST_SYNC, DST_ACCUM>, DST_SYNC, DST_ACCUM> {
    static void kernel(
        uint32_t idir, uint32_t i_end_phase, uint32_t i_start_phase, uint32_t i_end_step, uint32_t i_start_step) {
        calculate_bitonic_topk_phases_steps<APPROXIMATION_MODE, DST_ACCUM, STABLE_SORT>(
            idir, i_end_phase, i_start_phase, i_end_step, i_start_step);
    }

    static void init_kernel() { topk_init<APPROXIMATION_MODE>(); }
};

template <bool APPROXIMATION_MODE, bool IDIR, bool STABLE_SORT, DstSync DST_SYNC, bool DST_ACCUM>
struct TopkMerge
    : SfpuUnaryOp<TopkMerge<APPROXIMATION_MODE, IDIR, STABLE_SORT, DST_SYNC, DST_ACCUM>, DST_SYNC, DST_ACCUM> {
    static void kernel(uint32_t m_iter, uint32_t k) {
        calculate_bitonic_topk_merge<APPROXIMATION_MODE, DST_ACCUM, IDIR, STABLE_SORT>(m_iter, k);
    }

    static void init_kernel() { topk_init<APPROXIMATION_MODE>(); }
};

template <bool APPROXIMATION_MODE, bool STABLE_SORT, DstSync DST_SYNC, bool DST_ACCUM>
struct TopkRebuild
    : SfpuUnaryOp<TopkRebuild<APPROXIMATION_MODE, STABLE_SORT, DST_SYNC, DST_ACCUM>, DST_SYNC, DST_ACCUM> {
    static void kernel(uint32_t idir, uint32_t m_iter, uint32_t k, uint32_t logk, uint32_t skip_second) {
        calculate_bitonic_topk_rebuild<APPROXIMATION_MODE, DST_ACCUM, STABLE_SORT>(idir, m_iter, k, logk, skip_second);
    }

    static void init_kernel() { topk_init<APPROXIMATION_MODE>(); }
};

}  // namespace sfpu
}  // namespace ckernel

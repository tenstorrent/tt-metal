// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_topk.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::topk_local_sort, APPROXIMATE>(sfpu::topk_init<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_topk_local_sort(
    uint dst_index,
    int idir,
    int i_end_phase,
    int i_start_phase,
    int i_end_step,
    int i_start_step,
    int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_bitonic_topk_phases_steps<APPROXIMATE, is_fp32_dest_acc_en>,
        dst_index,
        vector_mode,
        idir,
        i_end_phase,
        i_start_phase,
        i_end_step,
        i_start_step);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, bool idir = false>
inline void llk_math_eltwise_unary_sfpu_topk_merge(
    uint dst_index, int m_iter, int k, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_bitonic_topk_merge<APPROXIMATE, is_fp32_dest_acc_en, idir>,
        dst_index,
        vector_mode,
        m_iter,
        k);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_topk_rebuild(
    uint dst_index,
    bool idir,
    int m_iter,
    int k,
    int logk,
    int skip_second,
    int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_bitonic_topk_rebuild<APPROXIMATE, is_fp32_dest_acc_en>,
        dst_index,
        vector_mode,
        idir,
        m_iter,
        k,
        logk,
        skip_second);
}

}  // namespace ckernel

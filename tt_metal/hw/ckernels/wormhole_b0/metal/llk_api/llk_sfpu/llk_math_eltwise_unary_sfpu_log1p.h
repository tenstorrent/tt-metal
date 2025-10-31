// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_log1p.h"
#include "ckernel_sfpu_log1p_exp.h"

namespace ckernel {

template <bool APPROXIMATE, bool FAST_APPROX>
inline void llk_math_eltwise_unary_sfpu_log1p_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log1p, APPROXIMATE>(sfpu::log1p_init<APPROXIMATE, FAST_APPROX>);
}

template <bool APPROXIMATE, bool FAST_APPROX>
inline void llk_math_eltwise_unary_sfpu_log1p(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_log1p<APPROXIMATE, FAST_APPROX>, dst_index, vector_mode);
}

template <bool APPROXIMATE, bool FAST_APPROX>
inline void llk_math_eltwise_unary_sfpu_log1p_exp_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log1p_exp, APPROXIMATE>(sfpu::log1p_exp_init<APPROXIMATE, FAST_APPROX>);
}

template <bool APPROXIMATE, bool FAST_APPROX, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_log1p_exp(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_log1p_exp<APPROXIMATE, FAST_APPROX, is_fp32_dest_acc_en, ITERATIONS>,
        dst_index,
        vector_mode);
}

}  // namespace ckernel

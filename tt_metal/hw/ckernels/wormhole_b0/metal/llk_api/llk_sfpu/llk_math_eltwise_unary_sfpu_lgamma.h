// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_math_eltwise_ternary_sfpu_params.h"
#include "ckernel_sfpu_lgamma.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_lgamma_stirling_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::lgamma, APPROXIMATE>(
        sfpu::lgamma_stirling_init<APPROXIMATE, is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_lgamma_stirling(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_lgamma_stirling<APPROXIMATE, is_fp32_dest_acc_en>, dst_index, vector_mode);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_binary_sfpu_lgamma_stirling_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::lgamma, APPROXIMATE>(
        sfpu::lgamma_stirling_init<APPROXIMATE, is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_binary_sfpu_lgamma_stirling(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_lgamma_stirling<APPROXIMATE, is_fp32_dest_acc_en>, dst_index0, dst_index1, odst, vector_mode);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_ternary_sfpu_lgamma_adjusted_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::lgamma>();
    // ckernel::sfpu::lgamma_adjusted_init<APPROXIMATE, is_fp32_dest_acc_en>();
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_ternary_sfpu_lgamma_adjusted(
    uint dst_index0, uint dst_index1, uint dst_index2, uint dst_index3, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_lgamma_adjusted<APPROXIMATE, is_fp32_dest_acc_en>,
        dst_index0,
        dst_index1,
        dst_index2,
        dst_index3,
        vector_mode);
}

}  // namespace ckernel

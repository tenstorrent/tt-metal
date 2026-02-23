// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_log.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_log_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log, APPROX_MODE>(sfpu::log_init<APPROX_MODE, is_fp32_dest_acc_en>);
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_log(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_log<APPROX_MODE, false, is_fp32_dest_acc_en>, dst_index, vector_mode, 0);
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_log_with_base_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log_with_base, APPROX_MODE>(
        sfpu::log_init<APPROX_MODE, is_fp32_dest_acc_en>);
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_log_with_base(
    uint dst_index, uint base_scale, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_log<APPROX_MODE, true, is_fp32_dest_acc_en>, dst_index, vector_mode, base_scale);
}

}  // namespace ckernel

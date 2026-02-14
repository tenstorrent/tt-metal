// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_unary_power.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_power_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::power, APPROXIMATE>(ckernel::sfpu::sfpu_unary_pow_init);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_power(
    uint dst_index, uint32_t exponent = 0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_power<APPROXIMATE, is_fp32_dest_acc_en, 8>, dst_index, vector_mode, exponent);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_power_iterative_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::power, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_power_iterative(
    uint dst_index, uint32_t exponent = 0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_power_iterative<APPROXIMATE, 8>, dst_index, vector_mode, exponent);
}
}  // namespace ckernel

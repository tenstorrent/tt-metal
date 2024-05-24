// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_tanh_derivative.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tanh_derivative_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanh_derivative, APPROXIMATE>(sfpu::tanh_derivative_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tanh_derivative(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE>(
        ckernel::sfpu::calculate_tanh_derivative<APPROXIMATE>,
        ckernel::sfpu::calculate_tanh_derivative<APPROXIMATE>,
        dst_index,
        vector_mode);
}

}  // namespace ckernel

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "ckernel_sfpu_gelu.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu(uint dst_index, int vector_mode = (int)VectorMode::RC, int param0=0) {
    constexpr int first_iterations = 1;
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE>
                                (ckernel::sfpu::calculate_gelu<APPROXIMATE, first_iterations>,
                                ckernel::sfpu::calculate_gelu<APPROXIMATE>,
                                dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::gelu, APPROXIMATE>(sfpu::gelu_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu_derivative(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    constexpr int first_iterations = 1;
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE>
                                (ckernel::sfpu::calculate_gelu_derivative<APPROXIMATE, first_iterations>,
                                ckernel::sfpu::calculate_gelu_derivative<APPROXIMATE>,
                                dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu_derivative_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_derivative, APPROXIMATE>(sfpu::gelu_derivative_init<APPROXIMATE>);
}

}

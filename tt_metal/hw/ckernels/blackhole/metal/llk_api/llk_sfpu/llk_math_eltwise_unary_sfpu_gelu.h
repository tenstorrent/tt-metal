// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_gelu.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_gelu(uint dst_index, int vector_mode = (int)VectorMode::RC, int param0 = 0) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_gelu<APPROXIMATE>, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::gelu, APPROXIMATE>(sfpu::gelu_init<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_gelu_derivative(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_gelu_derivative<APPROXIMATE>, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu_derivative_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_derivative, APPROXIMATE>(sfpu::gelu_derivative_init<APPROXIMATE>);
}

}  // namespace ckernel

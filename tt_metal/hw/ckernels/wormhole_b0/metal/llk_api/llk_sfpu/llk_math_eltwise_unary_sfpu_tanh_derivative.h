// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_tanh_derivative.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_tanh_derivative_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanh_derivative, APPROX_MODE>(sfpu::tanh_derivative_init<APPROX_MODE>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_tanh_derivative(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_tanh_derivative<APPROX_MODE>, dst_index, vector_mode);
}

}  // namespace ckernel

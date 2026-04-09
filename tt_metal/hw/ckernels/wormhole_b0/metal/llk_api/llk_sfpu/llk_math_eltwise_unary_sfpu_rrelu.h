// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_rrelu.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rrelu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>();
}

// Evaluation mode: slope = (lower + upper) / 2, pre-computed by host
template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu_eval(
    uint dst_index, int vector_mode = (int)VectorMode::RC, uint32_t slope_bits = 0) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu_eval<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, slope_bits);
}

// Training mode: lower and upper bounds for uniform random slope
template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu_train(
    uint dst_index, int vector_mode = (int)VectorMode::RC, uint32_t lower_bits = 0, uint32_t upper_bits = 0) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu_train<APPROXIMATE, ITERATIONS>,
        dst_index,
        vector_mode,
        lower_bits,
        upper_bits);
}

}  // namespace ckernel

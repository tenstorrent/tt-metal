// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_rrelu.h"

namespace ckernel {

// Eval mode init
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rrelu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

// Training mode init (seeds PRNG)
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rrelu_training_init(uint32_t seed) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
    ckernel::sfpu::_init_rrelu_training_(seed);
}

// Eval mode: single slope parameter
template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, int vector_mode = (int)VectorMode::RC, uint32_t slope = 0) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_rrelu_<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, slope);
}

// Training mode: lower and upper parameters
template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu_training(
    uint dst_index, int vector_mode = (int)VectorMode::RC, uint32_t lower = 0, uint32_t upper = 0) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_rrelu_training_<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, lower, upper);
}

}  // namespace ckernel

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "ckernel_sfpu_softsign.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardsigmoid_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid, APPROXIMATE>(
        ckernel::sfpu::_init_hardsigmoid_<APPROXIMATE>);
}

template <bool APPROXIMATE, ckernel::ActivationType ACTIVATION, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardsigmoid(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        static_cast<void (*)()>(ckernel::sfpu::_calculate_activation_<APPROXIMATE, ACTIVATION, ITERATIONS>),
        dst_index,
        vector_mode);
}

// softsign
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softsign_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softsign, APPROXIMATE>(ckernel::sfpu::init_softsign<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softsign(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_softsign<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_celu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::celu, APPROXIMATE>(
        ckernel::sfpu::_init_exponential_<APPROXIMATE, /*FAST_APPROX=*/APPROXIMATE, /*SCALE=*/p_sfpu::kCONST_1_FP16B>);
}

template <bool APPROXIMATE, ckernel::ActivationType ACTIVATION, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_celu(
    uint dst_index, uint32_t alpha, uint32_t alpha_recip, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        [](uint32_t alpha, uint32_t alpha_recip) {
            ckernel::sfpu::_calculate_activation_<APPROXIMATE, ACTIVATION, ITERATIONS>(alpha, alpha_recip);
        },
        dst_index,
        vector_mode,
        alpha,
        alpha_recip);
}
}  // namespace ckernel

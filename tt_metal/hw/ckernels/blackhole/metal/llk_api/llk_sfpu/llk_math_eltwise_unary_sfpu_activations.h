// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "ckernel_sfpu_softsign.h"
#include "ckernel_sfpu_softshrink.h"
#include "ckernel_sfpu_celu.h"

namespace ckernel {

// Hardsigmoid
template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_hardsigmoid_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid, APPROX_MODE>(
        ckernel::sfpu::_init_hardsigmoid_<APPROX_MODE>);
}

template <ckernel::ApproximationMode APPROX_MODE, ckernel::ActivationType ACTIVATION, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardsigmoid(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        static_cast<void (*)()>(ckernel::sfpu::_calculate_activation_<APPROX_MODE, ACTIVATION, ITERATIONS>),
        dst_index,
        vector_mode);
}

// softsign
template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_softsign_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softsign, APPROX_MODE>(ckernel::sfpu::init_softsign<APPROX_MODE>);
}

template <ckernel::ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softsign(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_softsign<APPROX_MODE, ITERATIONS>, dst_index, vector_mode);
}

// celu
template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_celu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::celu, APPROX_MODE>();
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_celu(
    uint dst_index, uint32_t alpha, uint32_t alpha_recip, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        [](uint32_t alpha, uint32_t alpha_recip) {
            ckernel::sfpu::calculate_celu<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>(alpha, alpha_recip);
        },
        dst_index,
        vector_mode,
        alpha,
        alpha_recip);
}

// softshrink
template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_softshrink_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softshrink, APPROX_MODE>();
}

template <ckernel::ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softshrink(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_softshrink<APPROX_MODE, ITERATIONS>, dst_index, vector_mode, param0);
}

}  // namespace ckernel

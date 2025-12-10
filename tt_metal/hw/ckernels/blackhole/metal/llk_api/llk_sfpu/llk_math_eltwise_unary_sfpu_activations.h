// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "ckernel_sfpu_softsign.h"
#include "ckernel_sfpu_softshrink.h"
#include "ckernel_sfpu_celu.h"
#include "llk_defs.h"

namespace ckernel {

// Hardsigmoid
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardsigmoid_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid, APPROXIMATE>(
        ckernel::sfpu::_init_hardsigmoid_<(APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise)>);
}

template <bool APPROXIMATE, ckernel::ActivationType ACTIVATION, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardsigmoid(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        static_cast<void (*)()>(ckernel::sfpu::_calculate_activation_<
                                (APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise),
                                ACTIVATION,
                                ITERATIONS>),
        dst_index,
        vector_mode);
}

// softsign
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softsign_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softsign, APPROXIMATE>(
        ckernel::sfpu::init_softsign<(APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise)>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softsign(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::
            calculate_softsign<(APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise), ITERATIONS>,
        dst_index,
        vector_mode);
}

// celu
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_celu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::celu, APPROXIMATE>();
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_celu(
    uint dst_index, uint32_t alpha, uint32_t alpha_recip, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        [](uint32_t alpha, uint32_t alpha_recip) {
            ckernel::sfpu::calculate_celu<
                (APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise),
                is_fp32_dest_acc_en,
                ITERATIONS>(alpha, alpha_recip);
        },
        dst_index,
        vector_mode,
        alpha,
        alpha_recip);
}

// softshrink
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softshrink_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softshrink, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softshrink(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::
            calculate_softshrink<(APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise), ITERATIONS>,
        dst_index,
        vector_mode,
        param0);
}

}  // namespace ckernel

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "ckernel_sfpu_softsign.h"
#include "ckernel_sfpu_softshrink.h"
#include "ckernel_sfpu_hardshrink.h"
#include "ckernel_sfpu_tanhshrink.h"
#include "ckernel_sfpu_celu.h"

namespace ckernel {

// Hardsigmoid
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
            ckernel::sfpu::calculate_celu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>(alpha, alpha_recip);
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
        ckernel::sfpu::calculate_softshrink<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0);
}

// tanhshrink
template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_unary_sfpu_tanhshrink_init() {
    if constexpr (APPROXIMATE) {
        ckernel::sfpu::tanh_init<APPROXIMATE, is_fp32_dest_acc_en>();
    } else {
        if constexpr (is_fp32_dest_acc_en) {
            ckernel::sfpu::sigmoid_init<false>();
        } else {
            sfpi::vConstFloatPrgm0 = 5.876733921468257904052734375e-3;
            sfpi::vConstFloatPrgm1 = -6.6649019718170166015625e-2;
            sfpi::vConstFloatPrgm2 = 0.281917631626129150390625;
        }
    }
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanhshrink, APPROXIMATE>();
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_tanhshrink(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_tanhshrink<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode);
}

// hardshrink
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardshrink_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardshrink, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardshrink(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_hardshrink<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0);
}

}  // namespace ckernel

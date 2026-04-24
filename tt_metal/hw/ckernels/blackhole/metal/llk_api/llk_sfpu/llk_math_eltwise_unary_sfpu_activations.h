// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "ckernel_sfpu_softsign.h"
#include "ckernel_sfpu_softshrink.h"
#include "ckernel_sfpu_hardshrink.h"
#include "ckernel_sfpu_celu.h"
#include "sfpu/ckernel_sfpu_activations.h"

namespace ckernel {

// Hardsigmoid
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardsigmoid_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid>(ckernel::sfpu::_init_hardsigmoid_<APPROXIMATE>);
}

template <bool APPROXIMATE, ckernel::ActivationType ACTIVATION, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardsigmoid(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_(
        static_cast<void (*)(std::uint32_t, std::uint32_t)>(
            ckernel::sfpu::_calculate_activation_<APPROXIMATE, ACTIVATION, ITERATIONS>),
        dst_index,
        dst_index,
        vector_mode);
}

// softsign
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softsign_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softsign>(ckernel::sfpu::init_softsign<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softsign(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::calculate_softsign<APPROXIMATE, ITERATIONS>, dst_index, dst_index, vector_mode);
}

// celu
inline void llk_math_eltwise_unary_sfpu_celu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::celu>();
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_celu(
    uint dst_index, uint32_t alpha, uint32_t alpha_recip, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_(
        [](std::uint32_t dst_index_in, std::uint32_t dst_index_out, std::uint32_t alpha, std::uint32_t alpha_recip) {
            ckernel::sfpu::calculate_celu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>(
                dst_index_in, dst_index_out, alpha, alpha_recip);
        },
        dst_index,
        dst_index,
        vector_mode,
        alpha,
        alpha_recip);
}

// softshrink
inline void llk_math_eltwise_unary_sfpu_softshrink_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softshrink>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softshrink(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::calculate_softshrink<APPROXIMATE, ITERATIONS>, dst_index, dst_index, vector_mode, param0);
}

// hardshrink
inline void llk_math_eltwise_unary_sfpu_hardshrink_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardshrink>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardshrink(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::calculate_hardshrink<APPROXIMATE, ITERATIONS>, dst_index, dst_index, vector_mode, param0);
}

}  // namespace ckernel

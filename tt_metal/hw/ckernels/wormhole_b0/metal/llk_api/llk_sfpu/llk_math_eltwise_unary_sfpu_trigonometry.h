// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_trigonometry.h"

namespace ckernel {

// New LLK SFPU APIs

// sine
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sine_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sine, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sine_op(uint dst_index) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_trig<SfpuType::sine, APPROXIMATE>, dst_index, (int)VectorMode::RC);
}

// cosine
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_cosine_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::cosine, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_cosine_op(uint dst_index) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_trig<SfpuType::cosine, APPROXIMATE>, dst_index, (int)VectorMode::RC);
}

// tangent
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tan_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::tan, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tan_op(uint dst_index) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_trig<SfpuType::tan, APPROXIMATE>, dst_index, (int)VectorMode::RC);
}

// asin
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_asin_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::asin, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_asin(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_asin<APPROXIMATE>, dst_index, vector_mode);
}

// acos
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_acos_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::acos, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_acos(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_acos<APPROXIMATE>, dst_index, vector_mode);
}

// acosh
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_acosh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::acosh, APPROXIMATE>(
        ckernel::sfpu::_init_inverse_hyperbolic_<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_acosh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_acosh_<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

// atan
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_atan_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::atan, APPROXIMATE>(sfpu::atan_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_atan(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_atan<APPROXIMATE>, dst_index, vector_mode);
}

// asinh
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_asinh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::asinh, APPROXIMATE>(
        ckernel::sfpu::_init_inverse_hyperbolic_<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_asinh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_asinh_<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);

}

// atanh
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_atanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(ckernel::sfpu::_init_atanh_<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_atanh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_atanh_<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel

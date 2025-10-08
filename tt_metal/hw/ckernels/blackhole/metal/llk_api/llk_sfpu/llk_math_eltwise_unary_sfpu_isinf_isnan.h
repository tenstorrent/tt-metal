// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel {

// isinf
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_isinf_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::isinf, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_isinf(uint dst_index) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_sfpu_isinf_isnan_<SfpuType::isinf, APPROXIMATE, 8>, dst_index, (int)VectorMode::RC);
}

// isposinf
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_isposinf_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::isposinf, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_isposinf(uint dst_index) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_sfpu_isinf_isnan_<SfpuType::isposinf, APPROXIMATE, 8>,
        dst_index,
        (int)VectorMode::RC);
}

// isneginf
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_isneginf_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::isneginf, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_isneginf(uint dst_index) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_sfpu_isinf_isnan_<SfpuType::isneginf, APPROXIMATE, 8>,
        dst_index,
        (int)VectorMode::RC);
}

// isnan
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_isnan_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::isnan, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_isnan(uint dst_index) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_sfpu_isinf_isnan_<SfpuType::isnan, APPROXIMATE, 8>, dst_index, (int)VectorMode::RC);
}

// isfinite
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_isfinite_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::isfinite, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_isfinite(uint dst_index) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_sfpu_isinf_isnan_<SfpuType::isfinite, APPROXIMATE, 8>,
        dst_index,
        (int)VectorMode::RC);
}

}  // namespace ckernel

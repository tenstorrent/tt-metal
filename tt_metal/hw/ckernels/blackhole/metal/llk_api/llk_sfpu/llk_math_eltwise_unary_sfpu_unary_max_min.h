// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_unary_max_min.h"

namespace ckernel {

// New LLK SFPU APIs

// Unary maximum
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_max_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_max, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_max(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_max_min<true, APPROXIMATE>, dst_index, vector_mode, param0);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_max_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_max_min_int32<true, APPROXIMATE>, dst_index, vector_mode, param0);
}

// Unary minimum
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_min_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_min, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_min(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_max_min<false, APPROXIMATE>, dst_index, vector_mode, param0);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_min_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_max_min_int32<false, APPROXIMATE>, dst_index, vector_mode, param0);
}

}  // namespace ckernel

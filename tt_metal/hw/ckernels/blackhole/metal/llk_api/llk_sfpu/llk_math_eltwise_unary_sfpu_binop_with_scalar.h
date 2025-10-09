// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_binop_with_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

template <bool APPROXIMATE, int binop_mode>
inline void llk_math_eltwise_unary_sfpu_binop_with_scalar(
    uint dst_index, uint32_t scalar, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_binop_with_scalar<APPROXIMATE, binop_mode, 8>, dst_index, vector_mode, scalar);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_binop_with_scalar_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_binop_with_scalar_add_int32(
    uint dst_index, uint32_t scalar, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_add_int32<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, scalar);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_binop_with_scalar_sub_int32(
    uint dst_index, uint32_t scalar, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_sub_int32<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, scalar);
}

}  // namespace ckernel

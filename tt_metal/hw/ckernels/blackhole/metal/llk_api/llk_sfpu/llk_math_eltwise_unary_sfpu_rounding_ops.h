// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rounding_op_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8, bool USE_FP32 = false>
inline void llk_math_eltwise_unary_sfpu_floor(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_floor_<APPROXIMATE, ITERATIONS, USE_FP32>, dst_index, vector_mode);
}

template <bool APPROXIMATE, int ITERATIONS = 8, bool USE_FP32 = true>
inline void llk_math_eltwise_unary_sfpu_floor_float32(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_floor_<APPROXIMATE, ITERATIONS, USE_FP32>, dst_index, vector_mode);
}

template <bool APPROXIMATE, int ITERATIONS = 8, bool USE_FP32 = false>
inline void llk_math_eltwise_unary_sfpu_ceil(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_ceil_<APPROXIMATE, ITERATIONS, USE_FP32>, dst_index, vector_mode);
}

template <bool APPROXIMATE, int ITERATIONS = 8, bool USE_FP32 = true>
inline void llk_math_eltwise_unary_sfpu_ceil_float32(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_ceil_<APPROXIMATE, ITERATIONS, USE_FP32>, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_trunc(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_trunc_<APPROXIMATE>, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_trunc_float32(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_trunc_<APPROXIMATE, true>, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_round(uint dst_index, int decimals, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_round_<APPROXIMATE>, dst_index, vector_mode, decimals);
}

}  // namespace ckernel

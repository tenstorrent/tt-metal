// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_unary_comp.h"

namespace ckernel {

// Unary Not equal
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_ne_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_ne, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_unary_ne_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_comp_unary_int<APPROXIMATE, SfpuType::unary_ne, ITERATIONS>,
        dst_index,
        vector_mode,
        param0);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_unary_ne(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_ne<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0);
}

// Unary equal
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_eq_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_eq, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_unary_eq_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_comp_unary_int<APPROXIMATE, SfpuType::unary_eq, ITERATIONS>,
        dst_index,
        vector_mode,
        param0);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_unary_eq(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_eq<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0);
}

// Unary greater than
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_gt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_gt, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_unary_gt_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_comp_unary_int_<APPROXIMATE, SfpuType::unary_gt, ITERATIONS>,
        dst_index,
        vector_mode,
        param0);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_unary_gt(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_gt<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0);
}

// Unary lesser than
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_lt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_lt, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_unary_lt_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_comp_unary_int_<APPROXIMATE, SfpuType::unary_lt, ITERATIONS>,
        dst_index,
        vector_mode,
        param0);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_unary_lt(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_lt<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0);
}

// Unary greater than or equal to
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_ge_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_ge, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_unary_ge_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_comp_unary_int_<APPROXIMATE, SfpuType::unary_ge, ITERATIONS>,
        dst_index,
        vector_mode,
        param0);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_unary_ge(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_ge<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0);
}

// Unary lesser than or equal to
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_le_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_le, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_unary_le_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_comp_unary_int_<APPROXIMATE, SfpuType::unary_le, ITERATIONS>,
        dst_index,
        vector_mode,
        param0);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_unary_le(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_le<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0);
}
}  // namespace ckernel

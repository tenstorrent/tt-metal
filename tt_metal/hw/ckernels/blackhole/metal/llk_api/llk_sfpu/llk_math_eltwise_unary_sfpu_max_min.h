// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_unary_max_min.h"

namespace ckernel {

// Unary maximum
template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_unary_max_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_max, APPROX_MODE>(sfpu::unary_max_min_init<true>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_unary_max(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_unary_max_min<true, APPROX_MODE>, dst_index, vector_mode, param0);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_unary_max_int32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_max_int32, APPROX_MODE>(
        sfpu::unary_max_min_int32_init<true, false>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_unary_max_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_unary_max_min_int32<true, false, APPROX_MODE>, dst_index, vector_mode, param0);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_unary_max_uint32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_max_uint32, APPROX_MODE>(
        sfpu::unary_max_min_int32_init<true, true>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_unary_max_uint32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_unary_max_min_int32<true, true, APPROX_MODE>, dst_index, vector_mode, param0);
}

// Unary minimum
template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_unary_min_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_min, APPROX_MODE>(sfpu::unary_max_min_init<false>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_unary_min(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_unary_max_min<false, APPROX_MODE>, dst_index, vector_mode, param0);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_unary_min_int32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_min_int32, APPROX_MODE>(
        sfpu::unary_max_min_int32_init<false, false>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_unary_min_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_unary_max_min_int32<false, false, APPROX_MODE>, dst_index, vector_mode, param0);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_unary_min_uint32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_min_uint32, APPROX_MODE>(
        sfpu::unary_max_min_int32_init<false, true>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_unary_min_uint32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_unary_max_min_int32<false, true, APPROX_MODE>, dst_index, vector_mode, param0);
}

}  // namespace ckernel

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary_max_min.h"

namespace ckernel {

// Binary maximum
template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_max_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::max, APPROX_MODE>(sfpu::binary_max_min_init<true>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_max(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_binary_max_min<true>, dst_index0, dst_index1, odst, vector_mode);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_max_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::max_int32, APPROX_MODE>(sfpu::binary_max_min_int32_init<true, false>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_max_int32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_binary_max_min_int32<true, false>, dst_index0, dst_index1, odst, vector_mode);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_max_uint32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::max_uint32, APPROX_MODE>(sfpu::binary_max_min_int32_init<true, true>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_max_uint32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_binary_max_min_int32<true, true>, dst_index0, dst_index1, odst, vector_mode);
}

// Binary minimum
template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_min_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::min, APPROX_MODE>(sfpu::binary_max_min_init<false>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_min(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_binary_max_min<false>, dst_index0, dst_index1, odst, vector_mode);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_min_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::min_int32, APPROX_MODE>(sfpu::binary_max_min_int32_init<false, false>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_min_int32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_binary_max_min_int32<false, false>, dst_index0, dst_index1, odst, vector_mode);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_min_uint32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::min_uint32, APPROX_MODE>(sfpu::binary_max_min_int32_init<false, true>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_min_uint32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_binary_max_min_int32<false, true>, dst_index0, dst_index1, odst, vector_mode);
}

}  // namespace ckernel

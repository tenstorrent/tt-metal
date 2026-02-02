// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_unary_max_min.h"

namespace ckernel {

// Unary maximum
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_max_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_max, APPROXIMATE>(sfpu::unary_max_min_init<true>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_max(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_max_min<true, APPROXIMATE>, dst_index, vector_mode, param0);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_max_int32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_max_int32, APPROXIMATE>(
        sfpu::unary_max_min_int32_init<true, false>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_max_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_max_min_int32<true, false, APPROXIMATE>, dst_index, vector_mode, param0);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_max_uint32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_max_uint32, APPROXIMATE>(
        sfpu::unary_max_min_int32_init<true, true>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_max_uint32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_max_min_int32<true, true, APPROXIMATE>, dst_index, vector_mode, param0);
}

// Unary minimum
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_min_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_min, APPROXIMATE>(sfpu::unary_max_min_init<false>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_min(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_max_min<false, APPROXIMATE>, dst_index, vector_mode, param0);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_min_int32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_min_int32, APPROXIMATE>(
        sfpu::unary_max_min_int32_init<false, false>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_min_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_max_min_int32<false, false, APPROXIMATE>, dst_index, vector_mode, param0);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_min_uint32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_min_uint32, APPROXIMATE>(
        sfpu::unary_max_min_int32_init<false, true>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_min_uint32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_max_min_int32<false, true, APPROXIMATE>, dst_index, vector_mode, param0);
}

}  // namespace ckernel

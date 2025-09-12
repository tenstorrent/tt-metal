// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary_comp.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_lt_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::lt, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_lt_int32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_comp_int32<APPROXIMATE, 8, SfpuType::lt>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_gt_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::gt, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_gt_int32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_comp_int32<APPROXIMATE, 8, SfpuType::gt>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

}  // namespace ckernel

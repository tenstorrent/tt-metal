// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_ternary_sfpu_params.h"
#include "ckernel_sfpu_where.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_where(
    uint dst_index0, uint dst_index1, uint dst_index2, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_ternary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_where<APPROXIMATE, 1>, dst_index0, dst_index1, dst_index2, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_where_fp32(
    uint dst_index0, uint dst_index1, uint dst_index2, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_ternary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_where<APPROXIMATE, 0>, dst_index0, dst_index1, dst_index2, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_where_int32(
    uint dst_index0, uint dst_index1, uint dst_index2, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_ternary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_where<APPROXIMATE, 2>, dst_index0, dst_index1, dst_index2, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_where_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::where>();
}
}  // namespace ckernel

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_cumsum.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_cumsum_row_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_cumsum_row(
    uint dst_index, uint first_tile, uint last_tile, int vector_mode = (int)VectorMode::RC_custom) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_cumsum_row<APPROXIMATE>, dst_index, vector_mode, first_tile, last_tile);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_cumsum_row_int_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_cumsum_row_int(
    uint dst_index, uint first_tile, uint last_tile, int vector_mode = (int)VectorMode::RC_custom) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_cumsum_row_int<APPROXIMATE>, dst_index, vector_mode, first_tile, last_tile);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_cumsum_row_flip_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_cumsum_row_flip(
    uint dst_index, uint mask_h, uint first_tile, uint last_tile, int vector_mode = (int)VectorMode::RC_custom) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_cumsum_row_flip<APPROXIMATE>, dst_index, vector_mode, mask_h, first_tile, last_tile);
}

}  // namespace ckernel

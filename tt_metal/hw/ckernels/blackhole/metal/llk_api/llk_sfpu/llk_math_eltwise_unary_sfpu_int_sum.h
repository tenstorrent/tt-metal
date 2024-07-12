// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_int_sum.h"

namespace ckernel {

enum SumIntDim{
    SUM_COL = 0,
    SUM_ROW
};

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sum_int_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::sum_int_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sum_int(uint dst_index, SumIntDim sum_int_dim) {
    if (sum_int_dim == SumIntDim::SUM_COL) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_sum_int_col<APPROXIMATE>, dst_index, (int)VectorMode::R);
    } else if (sum_int_dim == SumIntDim::SUM_ROW) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_sum_int_row<APPROXIMATE>, dst_index, (int)VectorMode::C);
    }
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_add_int(uint dst_index, uint dst_offset, int iterations, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::add_int<APPROXIMATE, 8>,
        dst_index,
        vector_mode,
        dst_offset
        );
}

}

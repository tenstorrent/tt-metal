// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_unary_comp.h"

namespace ckernel {

// New LLK SFPU APIs

//Unary Not equal
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_ne_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_ne, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_ne(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_ne<APPROXIMATE>,
        dst_index,
        vector_mode,
        param0);
}

//Unary greater than
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_gt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_gt, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_gt(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_gt<APPROXIMATE>,
        dst_index,
        vector_mode,
        param0);
}


//Unary lesser than
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_lt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_lt, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_lt(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_unary_lt<APPROXIMATE>,
        dst_index,
        vector_mode,
        param0);
}
}

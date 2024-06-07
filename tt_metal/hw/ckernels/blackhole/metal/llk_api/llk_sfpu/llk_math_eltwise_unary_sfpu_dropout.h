// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_dropout.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_dropout_init(uint seed = 0) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::dropout, APPROXIMATE>(sfpu::dropout_init<APPROXIMATE>, seed);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_dropout(uint dst_index, int vector_mode = (int)VectorMode::RC, int integer_dropout, int scale_factor) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_dropout<APPROXIMATE>,
        dst_index,
        vector_mode,
        integer_dropout,
        scale_factor);
}

}

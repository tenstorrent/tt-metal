// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_eltwise_unary_sfpu_common_includes.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "ckernel_sfpu_sqrt.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sqrt(uint dst_index, int vector_mode = VectorMode::RC) {
    constexpr int first_iterations = 1;
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE>
                                (ckernel::sfpu::calculate_sqrt<APPROXIMATE, first_iterations>,
                                ckernel::sfpu::calculate_sqrt<APPROXIMATE>,
                                dst_index, vector_mode);

}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sqrt_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(sfpu::sqrt_init<APPROXIMATE>);
}

}

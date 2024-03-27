// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "llk_math_eltwise_unary_sfpu_common_includes.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "ckernel_sfpu_i0.h"

namespace ckernel {

// New LLK SFPU APIs

//isinf
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_i0_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_i0_op(uint dst_index) {
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE>
                                (ckernel::sfpu::calculate_i0<APPROXIMATE,4>,
                                ckernel::sfpu::calculate_i0<APPROXIMATE,4>,
                                dst_index, VectorMode::RC);
}

}

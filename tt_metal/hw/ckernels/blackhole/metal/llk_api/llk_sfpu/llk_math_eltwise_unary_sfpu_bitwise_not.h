// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_bitwise_not.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_bitwise_not_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::bitwise_not, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_bitwise_not(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>
                                (ckernel::sfpu::calculate_bitwise_not<APPROXIMATE>,
                                dst_index, vector_mode, param0);
}
}

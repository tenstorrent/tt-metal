// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_exponential(uint dst_index, int vector_mode = (int)VectorMode::RC, int param0 = 0) {

	constexpr bool zero_negative = true;
    constexpr int first_iterations = 1;
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
      (ckernel::sfpu::calculate_exponential<APPROXIMATE, zero_negative, false, first_iterations>,
       ckernel::sfpu::calculate_exponential<APPROXIMATE, zero_negative>,
                                dst_index, vector_mode, param0);

}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_exponential_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, APPROXIMATE>(sfpu::exp_init<APPROXIMATE>);
}

}

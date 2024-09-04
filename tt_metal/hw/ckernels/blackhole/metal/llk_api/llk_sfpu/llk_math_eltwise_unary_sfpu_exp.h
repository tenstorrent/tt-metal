// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE, int ITERATIONS=8>
inline void llk_math_eltwise_unary_sfpu_exponential(uint dst_index, int vector_mode = (int)VectorMode::RC, int param0 = ITERATIONS, int param1 = 0) {

    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_exponential<APPROXIMATE>,
        dst_index,
        vector_mode,
        param0,
        param1);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_exponential_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, APPROXIMATE>(sfpu::exp_init<APPROXIMATE>);
}

}

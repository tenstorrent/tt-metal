// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel {

// New LLK SFPU APIs

template <
    bool APPROXIMATE,
    bool FAST_APPROX,
    bool SCALE_EN = false,
    bool SKIP_POSITIVE_CHECK = false,
    int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_exponential(
    uint dst_index, int vector_mode = (int)VectorMode::RC, int param0 = ITERATIONS, int param1 = 0x3F80) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_exponential<APPROXIMATE, FAST_APPROX, SCALE_EN, ITERATIONS, SKIP_POSITIVE_CHECK>,
        dst_index,
        vector_mode,
        param0,
        param1);
}

template <bool APPROXIMATE, bool FAST_APPROX, uint32_t scale = 0x3F800000>
inline void llk_math_eltwise_unary_sfpu_exponential_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, APPROXIMATE>(
        sfpu::exp_init<APPROXIMATE, FAST_APPROX, scale>);
}

}  // namespace ckernel

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_tanh_derivative.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tanh_derivative_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanh_derivative>(sfpu::tanh_derivative_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tanh_derivative(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanh_derivative, (APPROXIMATE), dst_index, vector_mode);
}

}  // namespace ckernel

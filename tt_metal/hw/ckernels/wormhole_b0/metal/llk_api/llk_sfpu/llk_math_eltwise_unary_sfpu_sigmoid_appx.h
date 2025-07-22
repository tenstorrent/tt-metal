// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_sigmoid_appx.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sigmoid_appx_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid_appx, APPROXIMATE>(sfpu::sigmoid_appx_init);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sigmoid_appx(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_sigmoid_appx, dst_index, vector_mode);
}

}  // namespace ckernel

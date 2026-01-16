// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_quickgelu_appx.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_quickgelu_appx_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::quickgelu_appx, APPROXIMATE>(sfpu::quickgelu_appx_init);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_quickgelu_appx(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_quickgelu_appx, dst_index, vector_mode);
}

}  // namespace ckernel

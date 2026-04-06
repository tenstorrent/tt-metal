// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_cbrt.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_cbrt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::cbrt, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_cbrt(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_cbrt<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel

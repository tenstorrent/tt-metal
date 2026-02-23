// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_sign.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_sign_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sign, APPROX_MODE>();
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_sign(
    uint dst_index, int vector_mode = (int)VectorMode::RC, uint exponent_size_8 = 1) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_sign<APPROX_MODE>, dst_index, vector_mode, exponent_size_8);
}

}  // namespace ckernel

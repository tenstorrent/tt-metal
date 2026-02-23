// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_clamp.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_clamp_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::clamp, APPROX_MODE>();
}

template <ckernel::ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_clamp(
    uint dst_index, uint min_val, uint max_val, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_clamp<APPROX_MODE, ITERATIONS>, dst_index, vector_mode, min_val, max_val);
}

template <ckernel::ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_clamp_int32(
    uint dst_index, uint min_val, uint max_val, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_clamp_int32<APPROX_MODE, ITERATIONS>, dst_index, vector_mode, min_val, max_val);
}

}  // namespace ckernel

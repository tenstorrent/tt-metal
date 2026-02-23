// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_threshold_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::threshold, APPROX_MODE>();
}

template <ckernel::ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_threshold(
    uint dst_index, uint32_t param0, uint32_t param1, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        static_cast<void (*)(uint32_t, uint32_t)>(ckernel::sfpu::_calculate_threshold_<APPROX_MODE, ITERATIONS>),
        dst_index,
        vector_mode,
        param0,
        param1);
}

}  // namespace ckernel

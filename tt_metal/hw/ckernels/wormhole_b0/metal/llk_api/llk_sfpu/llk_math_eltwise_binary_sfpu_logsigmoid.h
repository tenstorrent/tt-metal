// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_logsigmoid.h"

namespace ckernel {

// LogSigmoid operation using pre-computed scaled input and exponential
template <ckernel::ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_logsigmoid(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROX_MODE>(
        sfpu::calculate_logsigmoid<APPROX_MODE, ITERATIONS>, dst_index0, dst_index1, odst, vector_mode);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_logsigmoid_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX_MODE>();
}

}  // namespace ckernel

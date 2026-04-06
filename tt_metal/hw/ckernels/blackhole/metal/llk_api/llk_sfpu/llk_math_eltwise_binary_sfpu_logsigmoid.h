// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_logsigmoid.h"

namespace ckernel {

// LogSigmoid operation using pre-computed scaled input and exponential
template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_logsigmoid(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_logsigmoid<APPROXIMATE, ITERATIONS>, dst_index0, dst_index1, odst, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_logsigmoid_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

}  // namespace ckernel

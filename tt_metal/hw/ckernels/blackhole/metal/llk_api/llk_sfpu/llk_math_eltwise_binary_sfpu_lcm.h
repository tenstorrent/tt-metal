// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_lcm.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_lcm_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::lcm, APPROX_MODE>(sfpu::calculate_sfpu_lcm_init);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_lcm(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROX_MODE>(
        sfpu::calculate_sfpu_lcm, dst_index0, dst_index1, odst, vector_mode);
}

}  // namespace ckernel

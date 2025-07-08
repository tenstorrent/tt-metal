// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_gcd.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_gcd_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::gcd, APPROXIMATE>(
        sfpu::calculate_sfpu_gcd_init);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_gcd(
    uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(sfpu::calculate_sfpu_gcd, dst_index0, dst_index1, vector_mode);
}

}  // namespace ckernel

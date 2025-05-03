// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_sub_int32.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_sub_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE, bool SIGN_MAGNITUDE_FORMAT>
inline void llk_math_eltwise_binary_sfpu_sub_int32(
    uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_sub_int32<APPROXIMATE, SIGN_MAGNITUDE_FORMAT>, dst_index0, dst_index1, vector_mode);
}

}  // namespace ckernel

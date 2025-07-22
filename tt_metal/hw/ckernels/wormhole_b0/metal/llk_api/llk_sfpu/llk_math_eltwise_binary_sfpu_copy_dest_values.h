// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_copy_dest_values.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_math_eltwise_binary_sfpu_init.h"

namespace ckernel {

void llk_math_eltwise_binary_sfpu_copy_dest_values(
    uint32_t dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    constexpr bool APPROXIMATE = 0;
    // Note: this if is required due to the issue described in https://github.com/tenstorrent/tt-metal/issues/19442
    if (dst_index0 > dst_index1) {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            sfpu::copy_dest_value<APPROXIMATE, 1>, dst_index0, dst_index1, vector_mode);
    } else {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            sfpu::copy_dest_value<APPROXIMATE, 0>, dst_index0, dst_index1, vector_mode);
    }
}

inline void llk_math_eltwise_binary_sfpu_copy_dest_values_init() {
    constexpr bool APPROXIMATE = 0;
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::copy_dest_value_init);
}

}  // namespace ckernel

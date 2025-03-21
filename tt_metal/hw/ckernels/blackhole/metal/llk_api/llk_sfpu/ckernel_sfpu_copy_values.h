// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "llk_math_eltwise_binary_sfpu_params.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

namespace {

template <bool APPROXIMATION_MODE, bool IDST0_BIGGER, int ITERATIONS = 8>
void _copy_value_(const uint dst_offset) {
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 32;
        if constexpr (IDST0_BIGGER) {
            dst_reg[dst_offset * dst_tile_size] = dst_reg[0];
        } else {
            dst_reg[0] = dst_reg[dst_offset * dst_tile_size];
        }
        dst_reg++;
    }
}

void _copy_value_init_() {
    // No initialization required
}

}  // namespace

void llk_math_eltwise_binary_sfpu_copy_values(uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    constexpr bool APPROXIMATE = 0;
    // Note: this if is required due to the issue described in https://github.com/tenstorrent/tt-metal/issues/19442
    if (dst_index0 > dst_index1) {
        llk_math_eltwise_binary_sfpu_params<APPROXIMATE>(
            _copy_value_<APPROXIMATE, 1>, dst_index0, dst_index1, vector_mode);
    } else {
        llk_math_eltwise_binary_sfpu_params<APPROXIMATE>(
            _copy_value_<APPROXIMATE, 0>, dst_index0, dst_index1, vector_mode);
    }
}

inline void llk_math_eltwise_binary_sfpu_copy_values_init() {
    constexpr bool APPROXIMATE = 0;
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(ckernel::sfpu::_copy_value_init_);
}

}  // namespace sfpu
}  // namespace ckernel

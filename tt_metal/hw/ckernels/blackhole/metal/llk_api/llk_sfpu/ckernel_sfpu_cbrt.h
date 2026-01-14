// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include <limits>
#include "ckernel_sfpu_binary_pow.h"

namespace ckernel::sfpu {

constexpr float kOneThird = 1.0f / 3.0f;
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cube_root() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat input = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _sfpu_binary_power_<is_fp32_dest_acc_en>(sfpi::abs(input), kOneThird);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void cube_root_init() {
    sfpi::vConstFloatPrgm0 = 1.442695f;
    sfpi::vConstFloatPrgm1 = -127.0f;
    sfpi::vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN();
}

}  // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_power_iterative(const uint exponent) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        uint exp = exponent;
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = 1.0f;
        while (exp > 0) {
            if (exp & 1){
                result *= in;
            }
            in *= in;
            exp >>= 1;
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

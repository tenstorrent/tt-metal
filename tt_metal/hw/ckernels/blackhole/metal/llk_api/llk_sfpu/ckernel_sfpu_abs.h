// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::abs(v);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(1, 12, ADDR_MOD_7, 0);
        TTI_SFPABS(0, 1, 0, 0);
        TTI_SFPSTORE(0, 12, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

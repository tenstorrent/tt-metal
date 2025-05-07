// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool IS_MAX_OP = true, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_max_min(uint value) {

    // Load value param to lreg2
    TT_SFPLOADI(p_sfpu::LREG2, 10, value & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, value >> 16);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {

        // Load input to lreg0
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);

        // Copy value param to lreg1
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);

        // Swap and store maximum in lreg1, minimum in lreg0
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);

        if constexpr (IS_MAX_OP) {
            TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
        } else {
            TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
        }
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

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

template <bool IS_MAX_OP = true, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_max_min_int32(uint value) {
    int scalar = value;
    if (scalar < 0) {  // To convert from 2's complement to sign+magnitude
        scalar = -scalar;
        int res = 0x80000000 | (scalar & 0x7FFFFFFF);
        scalar = res;
    }

    // Load value param to lreg2
    _sfpu_load_imm32_(p_sfpu::LREG2, scalar);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load input tensor to lreg0
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_3, 0);

        // Copy value param to lreg2 to lreg1
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);

        // Swap and store maximum in lreg1, minimum in lreg0 (sign + magnitude format)
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);

        // Store the result
        if constexpr (IS_MAX_OP) {
            TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_3, 0);
        } else {
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_3, 0);
        }
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

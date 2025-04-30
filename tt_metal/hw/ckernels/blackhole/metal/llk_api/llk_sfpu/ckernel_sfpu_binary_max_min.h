// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(const uint dst_offset) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;

        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);                          // a
        TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, dst_offset * dst_tile_size);  // b

        // Swap and store maximum in lreg1, minimum in lreg0
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);
        TTI_SFPNOP;

        if constexpr (IS_MAX_OP) {
            TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);
        } else {
            TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
        }
        dst_reg++;
    }
}

template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(const uint dst_offset) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;

        constexpr auto INSTR_MOD_CAST = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;

        TTI_SFPLOAD(p_sfpu::LREG0, 12, ADDR_MOD_7, 0);                          // a
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG2, INSTR_MOD_CAST);
        TTI_SFPSETSGN(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);

        TT_SFPLOAD(p_sfpu::LREG1, 12, ADDR_MOD_7, dst_offset * dst_tile_size);  // b
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG2, INSTR_MOD_CAST);
        TTI_SFPSETSGN(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);

        // Swap and store maximum in lreg1, minimum in lreg0
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);
        TTI_SFPNOP;

        if constexpr (IS_MAX_OP) {
            TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG2, INSTR_MOD_CAST);
            TTI_SFPSETSGN(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
            TTI_SFPSTORE(p_sfpu::LREG1, 12, ADDR_MOD_7, 0);
        } else {
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG2, INSTR_MOD_CAST);
            TTI_SFPSETSGN(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            TTI_SFPSTORE(p_sfpu::LREG0, 12, ADDR_MOD_7, 0);
        }
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

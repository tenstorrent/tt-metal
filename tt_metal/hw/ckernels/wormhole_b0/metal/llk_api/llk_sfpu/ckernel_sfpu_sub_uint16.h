// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void sub_uint16(const uint dst_offset) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;
        // operand A - uint16
        TTI_SFPLOAD(p_sfpu::LREG1, LO16, ADDR_MOD_3, 0);
        // operand B - uint16
        TT_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_3, dst_offset * dst_tile_size);

        // Use 6 or LO16 as imod to convert operand B to 2's complement
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 6);
        TTI_SFPSTORE(p_sfpu::LREG0, LO16, ADDR_MOD_3, 0);

        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

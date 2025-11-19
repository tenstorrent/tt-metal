// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {

    constexpr uint dst_tile_size = 64;

    uint offset0 = dst_index_in0 * dst_tile_size;
    uint offset1 = dst_index_in1 * dst_tile_size;
    uint offset2 = dst_index_out * dst_tile_size;

    constexpr uint a0 = 0;
    constexpr uint b0 = 0;
    constexpr uint a1 = 1;
    constexpr uint b1 = 2;
    constexpr uint b2 = 3;
    constexpr uint c = 4;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // b0
        TT_SFPLOAD(b0, INT32, ADDR_MOD_7, offset1);
        // a1
        TT_SFPLOADMACRO((0 << 2) | (a1 & 3), INT32, ADDR_MOD_7, offset0 | (a1 >> 2));
        // b1
        TT_SFPLOADMACRO((1 << 2) | (b1 & 3), INT32, ADDR_MOD_7, offset1 | (b1 >> 2));
        // a0
        TT_SFPLOAD(a0, INT32, ADDR_MOD_7, offset0);
        // b2
        TT_SFPLOADMACRO((2 << 2) | (b2 & 3), INT32, ADDR_MOD_7, offset1 | (b2 >> 2));
        // c = mul24_hi(a0, b2)
        TTI_SFPMUL24(a0, b2, p_sfpu::LCONST_0, c, 1);
        // b1 = b1 + a1
        TTI_SFPIADD(0, a1, b1, SFPIADD_MOD1_CC_NONE);
        // c
        TT_SFPLOADMACRO((3 << 2) | (c & 3), INT32, ADDR_MOD_6, offset2 | (c >> 2));
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {
    constexpr uint b1 = 2;
    constexpr uint c = 4;

    TTI_SFPSHFT2(-23 & 0xfff, 0, 12, 6);
    TTI_SFPMUL24(0, 0, p_sfpu::LCONST_0, 13, 0);
    TTI_SFPSHFT(23, b1, 14, 1 | 4);
    TTI_SFPIADD(0, c, 15, SFPIADD_MOD1_CC_NONE);

    // Macro 0:
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0x80 | 0x00 | (1 << 3) | 5;
        constexpr uint round_bits = 0x80 | 0x00 | (0 << 3) | 4;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);

        TTI_SFPCONFIG(0, 4, 0);
    }
    // Macro 1:
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (4 << 3) | 7;
        constexpr uint mad_bits = 0x80 | 0x00 | (1 << 3) | 5;
        constexpr uint round_bits = 0x80 | 0x00 | (0 << 3) | 4;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);

        TTI_SFPCONFIG(0, 5, 0);
    }
    // Macro 2:
    {
        constexpr uint simple_bits = 0x80 | 0x40 | (4 << 3) | 7;
        constexpr uint mad_bits = 0x80 | 0x00 | (1 << 3) | 5;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 6, 1);
    }
    // Macro 3:
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (0 << 3) | 6;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);

        TTI_SFPCONFIG(0, 7, 0);
    }
    TTI_SFPCONFIG(0xff0, 8, 1);
}

}  // namespace ckernel::sfpu

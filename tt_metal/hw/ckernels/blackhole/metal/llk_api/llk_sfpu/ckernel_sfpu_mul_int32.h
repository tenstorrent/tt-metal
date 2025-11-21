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

    uint offset_in0 = dst_index_in0 * dst_tile_size;
    uint offset_in1 = dst_index_in1 * dst_tile_size;
    uint offset_out = dst_index_out * dst_tile_size;

    // This uses SFPLOADMACRO to achieve a throughput of 8 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.  Variables a0,
    // a1 are loaded from offset_in0; b0, b1 from offset_in1, and c from
    // offset_out.
    //
    //
    // t  | Load | Simple                 | MAD                     | Round                 | Store    |
    // -- | ---- | ---------------------- | ----------------------- | --------------------- | -------- |
    //  0 |  b0  |                        |                         |                       |          |
    //  1 | [a1] |                        |                         |                       |          |
    //  2 | [b1] |                        |                         | [a1] = shft2(a1, -23) |          |
    //  3 |  a0  |                        | [a1] = mul24_lo(b0, a1) | [b1] = shft2(b1, -23) |          |
    //  4 | [b2] |                        | [b1] = mul24_lo(a0, b1) |                       |          |
    //  5 |      |                        |  c   = mul24_hi(a0, b2) |                       |          |
    //  6 |      |  b1  = iadd(b1, a1)    | [b2] = mul24_lo(a0, b2) |                       |          |
    //  7 | [c]  | [b1] = iadd(b1, c)     |                         |                       |          |
    //  8 | .... | [c] = shft(b1, 23)     |                         |                       |          |
    //  9 | .... | [b2] L16 = iadd(b2, c) |                         |                       |          |
    // 10 | .... |                        |                         |                       | [c] L16  |
    //
    // In pseudocode, this is equivalent to:
    //
    // a1 = a >> 23
    // b1 = b >> 23
    // cross0 = mul24_lo(a1, b)
    // cross1 = mul24_lo(a, b1)
    // lo = mul24_lo(a, b)
    // hi = mul24_hi(a, b)
    // result = ((hi + cross0 + cross1) << 23) + lo

    constexpr uint a0 = 0;
    constexpr uint b0 = 0;
    constexpr uint a1 = 1;
    constexpr uint b1 = 2;
    constexpr uint b2 = 3;
    constexpr uint c = 4;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // b0
        TT_SFPLOAD(b0, INT32, ADDR_MOD_7, offset_in1);
        // a1
        TT_SFPLOADMACRO((0 << 2) | (a1 & 3), INT32, ADDR_MOD_7, offset_in0 | (a1 >> 2));
        // b1
        TT_SFPLOADMACRO((1 << 2) | (b1 & 3), INT32, ADDR_MOD_7, offset_in1 | (b1 >> 2));
        // a0
        TT_SFPLOAD(a0, INT32, ADDR_MOD_7, offset_in0);
        // b2
        TT_SFPLOADMACRO((2 << 2) | (b2 & 3), INT32, ADDR_MOD_7, offset_in1 | (b2 >> 2));
        // c = mul24_hi(a0, b2)
        TTI_SFPMUL24(a0, b2, p_sfpu::LCONST_0, c, sfpi::SFPMUL24_MOD1_UPPER);
        // b1 = b1 + a1
        TTI_SFPIADD(0, a1, b1, sfpi::SFPIADD_MOD1_CC_NONE);
        // c
        TT_SFPLOADMACRO((3 << 2) | (c & 3), INT32, ADDR_MOD_6, offset_out | (c >> 2));
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {
    constexpr uint b1 = 2;
    constexpr uint c = 4;

    // Load instruction templates.  This is more efficient than using
    // SFPCONFIG, but requires DISABLE_BACKDOOR_LOAD=false (the default).

    TTI_SFPSHFT2(-23 & 0xfff, 0, 12, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
    TTI_SFPMUL24(0, 0, p_sfpu::LCONST_0, 13, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPSHFT(23, b1, 14, 1 | 4);  // SFPSHFT_MOD1_ARG_IMM | SFPSHFT_MOD1_ARG_IMM_USE_VC
    TTI_SFPIADD(0, c, 15, sfpi::SFPIADD_MOD1_CC_NONE);

    // Configure macros.
    // See:
    // https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOADMACRO.md
    // For each of simple, MAD, round, store, a macro can use up to one
    // instruction template.  8 bits for each unit:
    // 0x80 means set VB=VD instead of VC=VD.
    // 0x40 means set VD=16 (only readable by scheduled SFPSTORE).
    // 3 bits for the delay.
    // 3 bits for template index (4+i), or 3 means SFPSTORE.

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
    // Misc: {
    //   StoreMod0: MOD0_FMT_SRCB,
    //   UsesLoadMod0ForStore: {1,1,1,1},
    //   UnitDelayKind: {1,1,1,1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0xff0, 8, 1);
}

}  // namespace ckernel::sfpu

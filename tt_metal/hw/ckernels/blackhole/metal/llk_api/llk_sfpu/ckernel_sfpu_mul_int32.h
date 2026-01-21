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

    constexpr uint a0 = p_sfpu::LREG0;
    constexpr uint b0 = p_sfpu::LREG0;
    constexpr uint a1 = p_sfpu::LREG1;
    constexpr uint b1 = p_sfpu::LREG2;
    constexpr uint b2 = p_sfpu::LREG3;
    constexpr uint c = p_sfpu::LREG4;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Note: SFPLOADMACRO requires VD to be split into:
        //   VDHi = VD >> 2 (1 bit)
        //   VDLo = VD & 3 (2 bits)
        // e.g. SFPLOADMACRO((m << 2) | VDLo, ..., offset | VDHi).

        // Load b0
        TT_SFPLOAD(b0, INT32, ADDR_MOD_7, offset_in1);
        // Macro 0, VD=a1
        TT_SFPLOADMACRO((0 << 2) | (a1 & 3), INT32, ADDR_MOD_7, offset_in0 | (a1 >> 2));
        // Macro 1, VD=b1
        TT_SFPLOADMACRO((1 << 2) | (b1 & 3), INT32, ADDR_MOD_7, offset_in1 | (b1 >> 2));
        // Load a0
        TT_SFPLOAD(a0, INT32, ADDR_MOD_7, offset_in0);
        // Macro 2, VD=b2
        TT_SFPLOADMACRO((2 << 2) | (b2 & 3), INT32, ADDR_MOD_7, offset_in1 | (b2 >> 2));
        // c = mul24_hi(a0, b2)
        TTI_SFPMUL24(a0, b2, p_sfpu::LCONST_0, c, sfpi::SFPMUL24_MOD1_UPPER);
        // b1 = b1 + a1
        TTI_SFPIADD(0, a1, b1, sfpi::SFPIADD_MOD1_CC_NONE);
        // Macro 3, VD=c
        TT_SFPLOADMACRO((3 << 2) | (c & 3), INT32, ADDR_MOD_6, offset_out | (c >> 2));
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {
    constexpr uint b1 = p_sfpu::LREG2;
    constexpr uint c = p_sfpu::LREG4;

    // Load instruction templates i=0-3.  This is more efficient than using
    // SFPCONFIG, but requires DISABLE_BACKDOOR_LOAD=false (the default).
    // Instruction template `i` is specified using `VD=12+i`.
    // Other register parameters (e.g. VA, VB, or VC) are set to 0 if they're
    // expected to be overridden by a macro.

    TTI_SFPSHFT2(-23 & 0xfff, 0, 12, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
    TTI_SFPMUL24(0, 0, p_sfpu::LCONST_0, 13, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPSHFT(23, b1, 14, 1 | 4);  // SFPSHFT_MOD1_ARG_IMM | SFPSHFT_MOD1_ARG_IMM_USE_VC
    TTI_SFPIADD(0, c, 15, sfpi::SFPIADD_MOD1_CC_NONE);

    // Configure macros.
    //
    // See:
    //   https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOADMACRO.md
    //
    // For each of simple, MAD, round, store, a macro can use up to one
    // instruction template.  This is packed into a 32-bit value per macro,
    // with 8 bits per unit:
    //
    //   (store << 24) | (round << 16) | (mad << 8) | simple.
    //
    // The 8 bits per unit (high to low) are:
    //
    // - bit 7: set means VB=VD instead of the default, VC=VD.
    // - bit 6: set means VD=16 (only readable by scheduled SFPSTORE).
    // - bits 5-3: delay (0-7).
    // - bits 2-0: template index (4+i), or 3 means SFPSTORE.
    //
    // The 32-bit value is then stored via SFPCONFIG to
    // LoadMacroConfig[lane].Sequence[m] where m means the macro index 0-3.

    // Macro 0:
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0x80 | 0x00 | (1 << 3) | (4 + 1);
        constexpr uint round_bits = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);

        // Configure Sequence[0], via VD=4+0
        TTI_SFPCONFIG(0, 4, 0);
    }
    // Macro 1:
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (4 << 3) | (4 + 3);
        constexpr uint mad_bits = 0x80 | 0x00 | (1 << 3) | (4 + 1);
        constexpr uint round_bits = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);

        // Configure Sequence[1], via VD=4+1
        TTI_SFPCONFIG(0, 4+1, 0);
    }
    // Macro 2:
    {
        constexpr uint simple_bits = 0x80 | 0x40 | (4 << 3) | (4 + 3);
        constexpr uint mad_bits = 0x80 | 0x00 | (1 << 3) | (4 + 1);

        // Configure Sequence[2], via VD=4+2
        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4+2, 1);
    }
    // Macro 3:
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 2);
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);

        // Configure Sequence[3], via VD=4+3
        TTI_SFPCONFIG(0, 4+3, 0);
    }
    // Misc: {
    //   StoreMod0: MOD0_FMT_SRCB,
    //   UsesLoadMod0ForStore: {1,1,1,1},
    //   UnitDelayKind: {1,1,1,1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0xff0, 8, 1);
}

}  // namespace ckernel::sfpu

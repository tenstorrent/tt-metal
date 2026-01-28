// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _mul_int_(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out)
{
    int offset0    = (dst_index_in0 * 32) << 1;
    int offset1    = (dst_index_in1 * 32) << 1;
    int offset_out = (dst_index_out * 32) << 1;

    constexpr int a0  = p_sfpu::LREG0;
    constexpr int b0  = p_sfpu::LREG1;
    constexpr int a1  = p_sfpu::LREG2;
    constexpr int b1  = p_sfpu::LREG3;
    constexpr int out = p_sfpu::LREG4;
    constexpr int tmp = p_sfpu::LREG5;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // This uses SFPLOADMACRO to achieve a throughput of 12 cycles per input row.
        //
        // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
        //
        // t  | Load  | Simple                | MAD                | Round       | Store     |
        // -- | ----- | --------------------- | ------------------ | ----------- | --------- |
        //  0 | [b1]  |                       |                    |             |           |
        //  1 | [a0]  |                       |                    | [b1] >>= 8  |           |
        //  2 | [b0]  | [b1] = cast(b1)       |                    |             |           |
        //  3 |       | a0 &= 0xff            |                    |             |           |
        //  4 |       | b0 &= 0xff            |                    |             |           |
        //  5 | [a1]  | [a0] = cast(a0)       |                    |             |           |
        //  6 | [out] | [b0] = cast(b0)       |                    |             |           |
        //  7 |       |                       | b1 = a0*b1 + 2**32 | [a1] >>= 8  |           |
        //  8 |       | [a1] = cast(a1)       | a0 = a0*b0 + 2**32 |             |           |
        //  9 |       |                       | b1 = a1*b0 + b1    |             |           |
        // 10 |       | tmp = exman(a0)       |                    |             |           |
        // 11 |       | out = exman(b1)       |                    |             |           |
        //  0 | ...   |                       |                    | [out] <<= 8 |           |
        //  1 | ...   | [out] L16 = tmp + out |                    |             |           |
        //  2 | ...   |                       |                    |             | [out] L16 |
        //
        // Split u16 inputs a and b into a = (a1 << 8) | a0; b = (b1 << 8) | b0,
        // where a0, a1, b0, b1 are u8.  Then cast to fp32, and calculate:
        //   lo  = a0*b0
        //   hi0 = a0*b1
        //   hi1 = a1*b0
        //
        // Observe that these are < 2**16, which lets us use the following trick:
        //
        // hi = a0*b1 + 2.0**23
        // lo = a0*b0 + 2.0**23
        // hi = a1*b0 + hi
        //
        // Adding the magic constant 2.0**23 lets us extract the integer value
        // via the raw mantissa bits.  The use of FMA means we can avoid doing
        // an integer addition later, and we can extract a single value hi,
        // instead of two values hi0 and hi1, saving two cycles.
        //
        // The final result will be lo + (hi << 8).

        TT_SFPLOADMACRO((0 << 2) | (b1 & 3), InstrModLoadStore::LO16, ADDR_MOD_3, offset1 | (b1 >> 2));
        TT_SFPLOADMACRO((1 << 2) | (a0 & 3), InstrModLoadStore::LO16, ADDR_MOD_3, offset0 | (a0 >> 2));
        TT_SFPLOADMACRO((1 << 2) | (b0 & 3), InstrModLoadStore::LO16, ADDR_MOD_3, offset1 | (b0 >> 2));

        TTI_SFPAND(0, p_sfpu::LREG12, a0, 0);
        TTI_SFPAND(0, p_sfpu::LREG12, b0, 0);

        TT_SFPLOADMACRO((2 << 2) | (a1 & 3), InstrModLoadStore::LO16, ADDR_MOD_3, offset0 | (a1 >> 2));
        TT_SFPLOADMACRO((3 << 2) | (out & 3), InstrModLoadStore::LO16, ADDR_MOD_2, offset_out | (out >> 2));

        // b1 (hi) = a0*b1 + 2.0**23
        TTI_SFPMAD(a0, b1, p_sfpu::LREG13, b1, 0);
        // a0 (lo) = a0*b0 + 2.0**23
        TTI_SFPMAD(a0, b0, p_sfpu::LREG13, a0, 0);
        // b1 (hi) += a1*b0
        TTI_SFPMAD(a1, b0, b1, b1, 0);

        // tmp = mantissa_bits(lo)
        TTI_SFPEXMAN(0, a0, tmp, sfpi::SFPEXMAN_MOD1_PAD9);
        // out = mantissa_bits(hi)
        TTI_SFPEXMAN(0, b1, out, sfpi::SFPEXMAN_MOD1_PAD9);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE>
inline void _init_mul_int_()
{
    sfpi::vConstIntPrgm0   = 0xff;      // LREG12
    sfpi::vConstFloatPrgm1 = 8388608.0; // LREG13

    constexpr int tmp = p_sfpu::LREG5;

    // InstructionTemplate[0]
    TTI_SFPSHFT2(-8 & 0xfff, 0, 12, 6); // SFPSHFT2_MOD1_SHFT_IMM

    // InstructionTemplate[1]
    TTI_SFPCAST(0, 13, 0);

    // InstructionTemplate[2]
    TTI_SFPSHFT2(8, 0, 14, 6); // SFPSHFT2_MOD1_SHFT_IMM

    // InstructionTemplate[3]
    TTI_SFPIADD(0, tmp, 15, sfpi::SFPIADD_MOD1_CC_NONE);

    // Macro 0
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (1 << 3) | (4 + 1);
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr uint store_bits  = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (3 << 3) | (4 + 1);
        constexpr uint mad_bits    = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1);
    }

    // Macro 2
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (2 << 3) | (4 + 1);
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0x80 | 0x00 | (1 << 3) | (4 + 0);
        constexpr uint store_bits  = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    // Macro 3
    {
        constexpr uint simple_bits = 0x80 | 0x40 | (6 << 3) | (4 + 3);
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0x80 | 0x00 | (5 << 3) | (4 + 2);
        constexpr uint store_bits  = 0x00 | 0x40 | (7 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    // Misc: {
    //   StoreMod0: DEFAULT,
    //   UsesLoadMod0ForStore: {1,1,1,1},
    //   UnitDelayKind: {1,1,1,1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0xff0, 8, 1);
}

} // namespace sfpu
} // namespace ckernel

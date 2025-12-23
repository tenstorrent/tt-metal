// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp16b_to_uint16_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, 0, 3, 0);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPLOADI(0, 0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 2, 0, 1, 6);
        TTI_SFPSTORE(1, 6, 3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp16b_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, 6, 3, 0);
        TTI_SFPCAST(0, 1, 0);
        TTI_SFP_STOCH_RND(0, 0, 3, 1, 2, 1);
        TTI_SFPSTORE(2, 0, 3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, 12, 3, 0);
        TTI_SFPCAST(0, 1, 0);
        TTI_SFP_STOCH_RND(0, 0, 3, 1, 2, 1);
        TTI_SFPSTORE(2, 0, 3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_int32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        // result = 0
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);

        // exp = in.Exp (LaneEnabled = exp >= 0)
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // result = INT_MIN
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);
        // exp -= 31 (LaneEnabled = exp < 31)
        TTI_SFPIADD(-31 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // exp += 8
        TTI_SFPIADD(8, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result = exman8(in) << (exp - 23)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);

        // LaneEnabled = in < 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        // result = -result (two's complement)
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        // result = 0
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);

        // LaneEnabled = in >= 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
        // exp = in.Exp (LaneEnabled = exp >= 0)
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // result = 0xffffffff
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
        // exp -= 32 (LaneEnabled = exp < 31)
        TTI_SFPIADD(-32 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // exp += 9
        TTI_SFPIADD(9, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result = exman8(in) << (exp - 23)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_fp16b_()
{
    // This uses SFPLOADMACRO to achieve a throughput of 3 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple          | MAD | Round      | Store   |
    // - | ---- | --------------- | --- | ---------- | ------- |
    // 0 |  [a] |                 |     |            |         |
    // 1 |  [b] |                 |     | [a] >>= 16 |         |
    // 2 |      | a &= 1          |     |            |         |
    // 0 |  ... | [b] += 0x7fff   |     |            |         |
    // 1 |  ... | [a] L16 = a + b |     |            | [a]     |
    // 2 |  ... |                 |     |            | [b] L16 |
    //
    // Note that [a] schedules a 32-bit store, writing all zeros except for the
    // LSB, which may be 0 or 1.  Then, [b] schedules a 16-bit store with
    // MOD0_FMT_BF16.  The zeros mean that even if rounding is applied by
    // packers, the result will be truncated.

    constexpr int b = p_sfpu::LREG2;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1;
        TT_SFPLOADMACRO((0 << 2) | (a & 3), 0, ADDR_MOD_3, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), 0, ADDR_MOD_2, b >> 2);
        TT_SFPAND(0, p_sfpu::LREG12, a, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp32_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, 6, 3, 0);
        TTI_SFPCAST(0, 1, 0);
        TTI_SFPSTORE(1, 3, 3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp32_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, 12, 3, 0);
        TTI_SFPCAST(0, 1, 0);
        TTI_SFPSTORE(1, 3, 3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp16b_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, 4, 3, 0);
        TTI_SFPSETSGN(0, 0, 1, 1);
        TTI_SFPCAST(1, 2, 0);
        TTI_SFP_STOCH_RND(0, 0, 4, 2, 3, 1);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPADDI(0x4f00, 3, 0); // 2^31
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(3, 0, 3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp32_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, 4, 3, 0);
        TTI_SFPSETSGN(0, 0, 1, 1);
        TTI_SFPCAST(1, 2, 0);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPADDI(0x4f00, 2, 0); // 2^31
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(2, 3, 3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_uint32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::LO16, ADDR_MOD_2, 0);
    }
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_uint16_()
{
    // Packer will read HI16 bits from DEST if DEST is in 32bit mode but pck is configured for uint16
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0); // Load 32bit datums from DEST
        TTI_SFPLOADI(p_sfpu::LREG1, 2, 0xFFFF);                              // Load Lo16 of LREG1 with 0xFFFF (65535)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 4);                                // If sign bit of LREG0 is 0, set CC
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);                     // Put smaller of LREG0 & 1 into LREG1
        TTI_SFPENCC(0, 0, 0, 0);                                             // Clear CC bits
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_3, 0); // Store output
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_uint16_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0); // Load 32bit datums from DEST
        TTI_SFPLOADI(p_sfpu::LREG1, 2, 0);                                   // Load Lo16 of LREG1 with 0
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);                     // Put smaller of LREG0 & 1 into LREG1
        TTI_SFPLOADI(p_sfpu::LREG2, 2, 0xFFFF);                              // Load Lo16 of LREG2 with 0xFFFF (65535)
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, 1);                     // Put smaller of LREG0 & 2 into LREG0
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_3, 0); // Store output
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_fp32_to_fp16b_()
{
    constexpr int b = p_sfpu::LREG2;

    sfpi::vConstIntPrgm0 = 1;
    sfpi::vConstIntPrgm1 = 0x7fff;

    // InstructionTemplate[0]
    TTI_SFPSHFT2(-16 & 0xfff, 0, 12, 6); // SFPSHFT2_MOD1_SHFT_IMM

    // InstructionTemplate[1]
    TTI_SFPIADD(0, p_sfpu::LREG13, 13, sfpi::SFPIADD_MOD1_CC_NONE);

    // InstructionTemplate[2]
    TTI_SFPIADD(0, b, 14, sfpi::SFPIADD_MOD1_CC_NONE);

    // Macro 0: [a]
    {
        constexpr uint simple_bits = 0x80 | 0x40 | (3 << 3) | (4 + 2);
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr uint store_bits  = 0x00 | 0x00 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1: [b]
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | (4 + 1);
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0;
        constexpr uint store_bits  = 0x00 | 0x40 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Misc: {
    //   StoreMod0: 2,
    //   UsesLoadMod0ForStore: {1,0},
    //   UnitDelayKind: {1,1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x312, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint16_to_uint32_()
{
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0;
        constexpr uint store_bits  = 0x00 | 0x00 | (0 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: InstrModLoadStore::INT32,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

} // namespace sfpu
} // namespace ckernel

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
inline void _calculate_typecast_fp32_to_uint16_()
{
    // This uses SFPLOADMACRO to achieve a throughput of 2 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple            | MAD | Round            | Store   |
    // - | ---- | ----------------- | --- | ---------------- | ------- |
    // 0 | [v]  |                   |     |                  |         |
    // 1 | nop  | [v] = max(v, 0.0) |     |                  |         |
    // 0 | ...  | (must be idle)    |     | (must be idle)   |         |
    // 1 | ...  |                   |     | [v] L16 = rnd(v) |         |
    // 0 | ...  |                   |     |                  | [v] L16 |

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1; // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_2, v >> 2);
        TTI_SFPNOP;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp16b_()
{
    // This uses SFPLOADMACRO to achieve a throughput of 1 cycle per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple        | MAD | Round            | Store   |
    // - | ---- | ------------- | --- | ---------------- | ------- |
    // 0 | [v]  |               |     |                  |         |
    // 0 | ...  | [v] = cast(v) |     |                  |         |
    // 0 | ...  |               |     | [v] L16 = rnd(v) |         |
    // 0 | ...  |               |     |                  | [v] L16 |

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1; // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_2, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_()
{
    // This uses SFPLOADMACRO to achieve a throughput of 4 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // Note: L0=0.0 and L1=-2**31.  The sign bit of abs(v) is stored in L7 and
    // used to pick L0 or L1 for SFPMAD's VA:
    //
    // - if sign bit is 0, then compute L0*1.0 + v = v
    // - if sign bit is 1, then compute L1*1.0 + v = -2**31 + 0.0 = -2**31
    //
    // t | Load | Simple             | MAD                 | Round            | Store   |
    // - | ---- | ------------------ | ------------------- | ---------------- | ------- |
    // 0 | [v]  |                    |                     |                  |         |
    // 1 |      | t = abs(v)         |                     |                  |         |
    // 2 |      |                    |                     | L7 = t >> 31     |         |
    // 3 |      | t = cast(t)        |                     |                  |         |
    // 0 | ...  | [v] = setsgn(t, v) |                     |                  |         |
    // 1 | ...  |                    | [v] = L[L7]*1.0 + v |                  |         |
    // 2 | ...  |                    |                     |                  |         |
    // 3 | ...  |                    |                     | [v] L16 = rnd(v) |         |
    // 0 | ...  |                    |                     |                  | [v] L16 |

    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00); // -2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between p_sfpu::LREG2 and p_sfpu::LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5); // SFPSHFT2_MOD1_SHFT_LREG
        TTI_SFPCAST(t, t, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
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
        int a = d & 1; // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
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
    // This uses SFPLOADMACRO to achieve a throughput of 1 cycle per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple            | MAD | Round | Store   |
    // - | ---- | ----------------- | --- | ----- | ------- |
    // 0 | [v]  |                   |     |       |         |
    // 0 | ...  | [v] L16 = cast(v) |     |       |         |
    // 0 | ...  |                   |     |       | [v] L16 |

    constexpr int v = p_sfpu::LREG0;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_2, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp32_()
{
    // This uses SFPLOADMACRO to achieve a throughput of 4 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // Note: L0=0.0 and L1=-2**31.  The sign bit of abs(v) is stored in L7 and
    // used to pick L0 or L1 for SFPMAD's VA:
    //
    // - if sign bit is 0, then compute L0*1.0 + v = v
    // - if sign bit is 1, then compute L1*1.0 + v = -2**31 + 0.0 = -2**31
    //
    // t | Load | Simple             | MAD                     | Round        | Store   |
    // - | ---- | ------------------ | ----------------------- | ------------ | ------- |
    // 0 | [v]  |                    |                         |              |         |
    // 1 |      | t = abs(v)         |                         |              |         |
    // 2 |      |                    |                         | L7 = t >> 31 |         |
    // 3 |      | t = cast(t)        |                         |              |         |
    // 0 | ...  | [v] = setsgn(t, v) |                         |              |         |
    // 1 | ...  |                    | [v] L16 = L[L7]*1.0 + v |              |         |
    // 2 | ...  |                    |                         |              |         |
    // 3 | ...  |                    |                         |              | [v] L16 |

    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00); // -2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between p_sfpu::LREG2 and p_sfpu::LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5); // SFPSHFT2_MOD1_SHFT_LREG
        TTI_SFPCAST(t, t, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp16b_()
{
    // This uses SFPLOADMACRO to achieve a throughput of 3 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // Note: L0=0.0 and L1=2**31.  The sign bit is stored in L7 and used to pick L0 or L1
    // for SFPMAD's VA:
    //
    // - if sign bit is 0, then compute L0*1.0 + v = v
    // - if sign bit is 1, then compute L1*1.0 + v = 2**31 + v
    //
    // t | Load | Simple             | MAD                     | Round            | Store   |
    // - | ---- | ------------------ | ----------------------- | ---------------- | ------- |
    // 0 | [v]  |                    |                         |                  |         |
    // 1 |      |                    |                         | L7 = v >> 31     |         |
    // 2 |      | v = setsgn(v, 0)   |                         |                  |         |
    // 0 | ...  | [v] = cast(v)      |                         |                  |         |
    // 1 | ...  |                    | [v] v = L[L7]*1.0 + v   |                  |         |
    // 2 | ...  |                    |                         |                  |         |
    // 0 | ...  |                    |                         | [v] L16 = rnd(v) |         |
    // 1 | ...  |                    |                         |                  | [v] L16 |

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00); // 2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between p_sfpu::LREG2 and p_sfpu::LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPSHFT2(v, p_sfpu::LREG12, p_sfpu::LREG7, 5); // SFPSHFT2_MOD1_SHFT_LREG
        TT_SFPSETSGN(0, v, v, 1);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp32_()
{
    // This uses SFPLOADMACRO to achieve a throughput of 3 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // Note: L0=0.0 and L1=2**31.  The sign bit is stored in L7 and used to pick L0 or L1
    // for SFPMAD's VA:
    //
    // - if sign bit is 0, then compute L0*1.0 + v = v
    // - if sign bit is 1, then compute L1*1.0 + v = 2**31 + v
    //
    // t | Load | Simple             | MAD                     | Round       | Store    |
    // - | ---- | ------------------ | ----------------------- | ----------- | -------- |
    // 0 | [a]  |                    |                         |             |          |
    // 1 | [b]  | [a] = setsgn(a, 0) |                         |             |          |
    // 2 | [L7] | [b] = cast(a)      |                         |             |          |
    // 0 | ...  |                    |                         | [L7] >>= 31 |          |
    // 1 | ...  |                    | [b] L16 = L[L7]*1.0 + b |             |          |
    // 2 | ...  |                    |                         |             |          |
    // 0 | ...  |                    |                         |             | [L7] L16 |

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00); // 2**31

    constexpr int a  = p_sfpu::LREG2;
    constexpr int b  = p_sfpu::LREG3;
    constexpr int L7 = p_sfpu::LREG7;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_3, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_3, b >> 2);
        TTI_SFPLOADMACRO((2 << 2) | (L7 & 3), InstrModLoadStore::INT32, ADDR_MOD_2, L7 >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_uint32_()
{
    // This uses SFPLOADMACRO to achieve a throughput of 1 cycle per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple | MAD | Round | Store |
    // - | ---- | ------ | --- | ----- | ----- |
    // 0 | [v]  |        |     |       |       |
    // 0 | ...  |        |     |       | [v]   |

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
    // This uses SFPLOADMACRO to achieve a throughput of 2 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple          | MAD | Round      | Store   |
    // - | ---- | --------------- | --- | ---------- | ------- |
    // 0 | [a]  |                 |     |            |         | Load high 16 bits, i.e. a = value >> 16
    // 1 | ...  | [a] = 0 - a     |     |            |         |
    // 0 | ...  |                 |     | [a] >>= 16 |         |
    // 1 | [b]  |                 |     |            |         |
    // 0 | ...  | [b] L16 = b | a |     |            |         |
    // 1 | ...  |                 |     |            | [b] L16 |

    constexpr int a = p_sfpu::LREG0;
    constexpr int b = p_sfpu::LREG2;

#pragma GCC unroll 9
    for (int d = 0; d < ITERATIONS + 1; d++)
    {
        int a          = d & 1;       // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        int macroIndex = 1 + (d & 1); // alternate between macros 1 and 2
        if (d < ITERATIONS)
        {
            TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::LO16, ADDR_MOD_2, a >> 2);
        }
        else
        {
            TTI_SFPNOP;
        }
        if (d == 0)
        {
            TTI_SFPNOP;
        }
        else if (d < ITERATIONS)
        {
            TTI_SFPLOADMACRO((macroIndex << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_3, (-4 & 0x3ff) | (b >> 2));
        }
        else
        {
            TTI_SFPLOADMACRO((macroIndex << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_2, (-2 & 0x3ff) | (b >> 2));
        }
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_uint16_()
{
    // This uses SFPLOADMACRO to achieve a throughput of 3 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple            | MAD | Round            | Store   |
    // - | ---- | ----------------- | --- | ---------------- | ------- |
    // 0 | [a]  |                   |     |                  |         |
    // 1 |      | a = cast_fp32(a)  |     |                  |         |
    // 2 | nop  | [a] = max(0.0, a) |     |                  |         |
    // 0 | ...  | (must be idle)    |     |                  |         |
    // 1 | ...  |                   |     | [a] L16 = rnd(a) |         |
    // 2 | ...  |                   |     |                  | [a] L16 |
    //
    // Simple/Round sub-units can be used simultaneously if one has VD=16 and
    // the other VD!=16.  The following steps clamp the input value to 0-65535:
    //
    // a = cast_fp32(a); this allows us to use SFPSTOCHRND later to convert to uint16, clamping to 65535.
    // swap_minmax(0.0, a); since SFPSTOCHRND takes the absolute value before clamping, we use SFPSWAP to clamp negative values to 0.0.
    // L16 = rnd(a); finally, we use SFPSTOCHRND to clamp large values to 65535, using VD=16.

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1; // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_2, a >> 2);
        TT_SFPCAST(a, a, 0);
        TTI_SFPNOP;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
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
    //   StoreMod0: FP16B,
    //   UsesLoadMod0ForStore: {1,0},
    //   UnitDelayKind: {1,1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x310 | InstrModLoadStore::FP16B, 8, 1);
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
    //   StoreMod0: INT32,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint32_to_fp32_()
{
    sfpi::vConstIntPrgm0 = -31;

    constexpr int a = p_sfpu::LREG2;

    // InstructionTemplate[0]
    TTI_SFPSETSGN(0, 0, 12, 1); // SFPSETSGN_MOD1_ARG_IMM

    // InstructionTemplate[1]
    TTI_SFPCAST(a, 13, 0);

    // InstructionTemplate[2]
    TTI_SFPSHFT2(0, p_sfpu::LREG12, 14, 5); // SFPSHFT2_MOD1_SHFT_LREG

    // InstructionTemplate[3]
    TTI_SFPMAD(0, p_sfpu::LCONST_1, 0, 15, 4); // SFPMAD_MOD1_INDIRECT_VA

    // Macro 0: [a]
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr uint mad_bits    = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 0, 1);
    }
    // Macro 1: [b]
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 1);
        constexpr uint mad_bits    = 0x00 | 0x40 | (2 << 3) | (4 + 3);

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1);
    }
    // Macro 2: [L7]
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0x80 | 0x00 | (0 << 3) | (4 + 2);
        constexpr uint store_bits  = 0x00 | 0x40 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    // Misc: {
    //   StoreMod0: FP32,
    //   UsesLoadMod0ForStore: {0,0,0},
    //   UnitDelayKind: {1,1,1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x700 | InstrModLoadStore::FP32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_int32_to_fp32_()
{
    constexpr int t = p_sfpu::LREG4;

    sfpi::vConstIntPrgm0 = -31;

    // InstructionTemplate[0]
    TTI_SFPSETSGN(0, t, 12, 0);

    // InstructionTemplate[1]
    TTI_SFPMAD(0, p_sfpu::LCONST_1, 0, 13, 4); // SFPMAD_MOD1_INDIRECT_VA

    // Macro 0: [v]
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | (4 + 0);
        constexpr uint mad_bits    = 0x00 | 0x40 | (4 << 3) | (4 + 1);
        constexpr uint round_bits  = 0;
        constexpr uint store_bits  = 0x00 | 0x40 | (6 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: FP32,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::FP32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_int32_to_fp16b_()
{
    constexpr int t = p_sfpu::LREG4;

    sfpi::vConstIntPrgm0 = -31;

    // InstructionTemplate[0]
    TTI_SFPSETSGN(0, t, 12, 0);

    // InstructionTemplate[1]
    TTI_SFPMAD(0, p_sfpu::LCONST_1, 0, 13, 4); // SFPMAD_MOD1_INDIRECT_VA

    // InstructionTemplate[2]
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 1); // SFPSTOCHRND_MOD1_FP32_TO_FP16B

    // Macro 0: [v]
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | (4 + 0);
        constexpr uint mad_bits    = 0x00 | 0x00 | (4 << 3) | (4 + 1);
        constexpr uint round_bits  = 0x00 | 0x40 | (6 << 3) | (4 + 2);
        constexpr uint store_bits  = 0x00 | 0x40 | (7 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: DEFAULT,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::DEFAULT, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint16_to_fp32_()
{
    // InstructionTemplate[0]
    TTI_SFPCAST(0, 12, 0);

    // Macro 0
    {
        constexpr uint simple_bits = 0x00 | 0x40 | (0 << 3) | (4 + 0);
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0;
        constexpr uint store_bits  = 0x00 | 0x40 | (1 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: FP32,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::FP32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint16_to_fp16b_()
{
    // InstructionTemplate[0]
    TTI_SFPCAST(0, 12, 0);

    // InstructionTemplate[1]
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 13, 1); // SFPSTOCHRND_MOD1_FP32_TO_FP16B

    // Macro 0
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0x00 | 0x40 | (1 << 3) | (4 + 1);
        constexpr uint store_bits  = 0x00 | 0x40 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: DEFAULT,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::DEFAULT, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint32_to_fp16b_()
{
    // InstructionTemplate[0]
    TTI_SFPCAST(0, 12, 0);

    // InstructionTemplate[1]
    TTI_SFPMAD(0, p_sfpu::LCONST_1, 0, 13, 4); // SFPMAD_MOD1_INDIRECT_VA

    // InstructionTemplate[2]
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 1); // SFPSTOCHRND_MOD1_FP32_TO_FP16B

    // Macro 0
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (2 << 3) | (4 + 0);
        constexpr uint mad_bits    = 0x00 | 0x00 | (3 << 3) | (4 + 1);
        constexpr uint round_bits  = 0x00 | 0x40 | (5 << 3) | (4 + 2);
        constexpr uint store_bits  = 0x00 | 0x40 | (6 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: DEFAULT,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::DEFAULT, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_fp32_to_uint16_()
{
    // InstructionTemplate[0]
    TTI_SFPSWAP(0, p_sfpu::LCONST_0, 12, 0xf); // L[VD] = max(0, L[VD])

    // InstructionTemplate[1]
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 13, 6); // SFPSTOCHRND_MOD1_FP32_TO_UINT16

    // Macro 0
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0x00 | 0x40 | (2 << 3) | (4 + 1);
        constexpr uint store_bits  = 0x00 | 0x40 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: LO16,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::LO16, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint32_to_uint16_()
{
    constexpr int a0 = p_sfpu::LREG0;
    constexpr int a1 = p_sfpu::LREG1;

    // InstructionTemplate[0]
    TTI_SFPIADD(0, p_sfpu::LCONST_0, 12, sfpi::SFPIADD_MOD1_CC_NONE | sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);

    // InstructionTemplate[1]
    TTI_SFPSHFT2(-16 & 0xfff, 0, 13, 6); // SFPSHFT2_MOD1_SHFT_IMM

    // InstructionTemplate[2]
    TTI_SFPOR(0, a1, 14, 0);

    // InstructionTemplate[3]
    TTI_SFPOR(0, a0, 15, 0);

    // Macro 0
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0x80 | 0x00 | (1 << 3) | (4 + 1);
        constexpr uint store_bits  = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1
    {
        constexpr uint simple_bits = 0x80 | 0x40 | (0 << 3) | (4 + 2);
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0;
        constexpr uint store_bits  = 0x00 | 0x40 | (1 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Macro 2
    {
        constexpr uint simple_bits = 0x80 | 0x40 | (0 << 3) | (4 + 3);
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0;
        constexpr uint store_bits  = 0x00 | 0x40 | (1 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    // Misc: {
    //   StoreMod0: LO16,
    //   UsesLoadMod0ForStore: {0,0,0},
    //   UnitDelayKind: {1,1,1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x700 | InstrModLoadStore::LO16, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_int32_to_uint16_()
{
    // InstructionTemplate[0]
    TTI_SFPSWAP(0, p_sfpu::LCONST_0, 12, 0xf); // L[VD] = max(0, L[VD])

    // InstructionTemplate[1]
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 13, 6); // SFPSTOCHRND_MOD1_FP32_TO_UINT16

    // Macro 0
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | (4 + 0);
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0x00 | 0x40 | (3 << 3) | (4 + 1);
        constexpr uint store_bits  = 0x00 | 0x40 | (4 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: LO16,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::LO16, 8, 1);
}

} // namespace sfpu
} // namespace ckernel

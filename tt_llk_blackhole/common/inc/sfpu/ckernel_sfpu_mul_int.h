// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _mul_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    // This uses SFPLOADMACRO to achieve a throughput of 1, 2, or 3 cycles per
    // input row:
    //
    // - dst_index_in0 == dst_index_in1 == dst_index_out: 1 cycle
    // - dst_index_in0 == dst_index_out or dst_index_in1 == dst_index_out: 2 cycles
    // - otherwise: 3 cycles
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple | MAD                      | Round | Store   |
    // - | ---- | ------ | ------------------------ | ----- | ------- |
    // 0 | [a]  |        |                          |       |         |
    // 0 | ...  |        | [a] L16 = mul24_lo(a, a) |       |         |
    // 1 | ...  |        |                          |       |         |
    // 0 | ...  |        |                          |       | [a] L16 |
    //
    // t | Load | Simple | MAD                      | Round | Store   |
    // - | ---- | ------ | ------------------------ | ----- | ------- |
    // 0 |  a   |        |                          |       |         |
    // 1 | [b]  |        |                          |       |         |
    // 0 | ...  |        | [b] L16 = mul24_lo(a, b) |       |         |
    // 1 | ...  |        |                          |       |         |
    // 0 | ...  |        |                          |       | [b] L16 |
    //
    // t | Load | Simple | MAD                      | Round | Store   |
    // - | ---- | ------ | ------------------------ | ----- | ------- |
    // 0 |  a   |        |                          |       |         |
    // 1 | [b]  |        |                          |       |         |
    // 2 |  b   |        |                          |       |         |
    // 0 | ...  |        | [b] L16 = mul24_lo(a, b) |       |         |
    // 1 | ...  |        |                          |       |         |
    // 2 | ...  |        |                          |       | [b] L16 |

    int offset0 = (dst_index_in0 * 32) << 1;
    int offset1 = (dst_index_in1 * 32) << 1;

    constexpr int a = p_sfpu::LREG0;
    constexpr int b = p_sfpu::LREG1;

    if (dst_index_out == dst_index_in0 && dst_index_out == dst_index_in1)
    {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            TT_SFPLOADMACRO((0 << 2) | (a & 3), LO16, ADDR_MOD_6, offset1 | (a >> 2));
        }
    }
    else if (dst_index_out == dst_index_in0)
    {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            TT_SFPLOAD(a, LO16, ADDR_MOD_7, offset1);
            TT_SFPLOADMACRO((0 << 2) | (b & 3), LO16, ADDR_MOD_6, offset0 | (b >> 2));
        }
    }
    else if (dst_index_out == dst_index_in1)
    {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            TT_SFPLOAD(a, LO16, ADDR_MOD_7, offset0);
            TT_SFPLOADMACRO((0 << 2) | (b & 3), LO16, ADDR_MOD_6, offset1 | (b >> 2));
        }
    }
    else
    {
        int offset2 = (dst_index_out * 32) << 1;

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            TT_SFPLOAD(a, LO16, ADDR_MOD_7, offset0);
            TT_SFPLOADMACRO((1 << 2) | (b & 3), LO16, ADDR_MOD_7, offset2 | (b >> 2));
            TT_SFPLOAD(b, LO16, ADDR_MOD_6, offset1);
        }
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE>
inline void _init_mul_int_()
{
    // InstructionTemplate[0]
    TTI_SFPMUL24(p_sfpu::LREG0, 0, p_sfpu::LCONST_0, 12, 0);

    // Macro 0
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits    = 0x80 | 0x40 | (0 << 3) | 4;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1:
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits    = 0x80 | 0x40 | (1 << 3) | 4;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Misc: {
    //   StoreMod0: DEFAULT,
    //   UsesLoadMod0ForStore: {1,1},
    //   UnitDelayKind: {1,1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x330, 8, 1);
}

} // namespace sfpu
} // namespace ckernel

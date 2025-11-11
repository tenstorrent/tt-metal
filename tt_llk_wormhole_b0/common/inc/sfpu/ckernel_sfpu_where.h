// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE, DataFormat data_format, int ITERATIONS>
inline void _calculate_where_(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_in2, const uint dst_index_out)
{
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b || data_format == DataFormat::Int32 || data_format == DataFormat::UInt32,
        "Unsupported data format for _calculate_where_(). Only Float32, Int32, UInt32, and Float16_b are allowed.");

    int offset0 = (dst_index_in0 * 32) << 1;
    int offset1 = (dst_index_in1 * 32) << 1;
    int offset2 = (dst_index_in2 * 32) << 1;

    constexpr uint mod0 = data_format == DataFormat::Float16_b ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;

    if (dst_index_out == dst_index_in0)
    {
        // We use macros 0 and 2 to schedule the following, which achieves 3 cycles per input row of 32 values:

        // Load Unit               | Simple Unit                    | Store Unit
        // SFPLOAD L0=Dst[offset0] |                                |
        // SFPLOAD L0=Dst[offset1] | SFPSETCC LaneEnabled=(L0 EQ 0) |
        // SFPLOAD L0=Dst[offset2] | SFPENCC (LaneEnabled=true)     |
        // (next SFPLOAD L0)       |                                | SFPSTORE Dst[offset0]=L0

        lltt::record(0, 3);
        TT_SFPLOADMACRO((0 << 2), mod0, ADDR_MOD_3, offset0);
        TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_3, offset1);
        TT_SFPLOAD(0, mod0, ADDR_MOD_2, offset2);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 3);
        }
    }
    else
    {
        // We use macros 1 and 2 to schedule the following, which achieves 4 cycles per input row of 32 values:

        // Load Unit               | Simple Unit                    | Store Unit
        // SFPLOAD L0=Dst[offset0] |                                |
        // SFPLOAD L0=Dst[offset1] | SFPSETCC LaneEnabled=(L0 EQ 0) |
        // SFPLOAD L0=Dst[offset2] | SFPENCC (LaneEnabled=true)     |
        // -                       |                                | SFPSTORE Dst[offset3]=L0
        // (next SFPLOAD L0)       |                                |

        int offset3 = (dst_index_out * 32) << 1;

        lltt::record(0, 4);
        TT_SFPLOADMACRO((1 << 2), mod0, ADDR_MOD_3, offset0);
        TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_3, offset1);
        TT_SFPLOAD(0, mod0, ADDR_MOD_3, offset2);
        TT_SFPSTORE(0, mod0, ADDR_MOD_2, offset3);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 4);
        }
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_where_()
{
    // InstructionTemplate[0]
    TTI_SFPSETCC(0, 0, 12, 6); // SFPSETCC_MOD1_LREG_EQ0

    // InstructionTemplate[1]
    TTI_SFPENCC(0, 0, 13, 0);

    // Macro 0: special case handling for where(a, b, c, a), i.e. write the output to the first input.
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (0 << 3) | 4;
        constexpr uint mad_bits    = 0;
        constexpr uint round_bits  = 0;
        constexpr uint store_bits  = 0x00 | 0x00 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1: otherwise, handle where(a, b, c, d).
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (0 << 3) | 4;
        constexpr uint mad_bits    = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1);
    }

    // Macro 2:
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (0 << 3) | 5;
        constexpr uint mad_bits    = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1);
    }

    // Misc: {UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1} for all macros.
    TTI_SFPCONFIG(0x770, 8, 1);
}

} // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, bool SIGN_MAGNITUDE_FORMAT, int ITERATIONS>
inline void _sub_int32_(const uint dst_offset)
{
    // Operand A is input1 (int32)
    // Operand B is input2 (int32)
    // Output is int32

    // Modifies LOAD/STORE to do INT32 sign-magnitude to 2's complement conversion, however
    // in Blackhole this has no effect and format remains in original format.
    constexpr auto INSTR_MOD_LOAD_STORE = InstrModLoadStore::INT32_2S_COMP;

    // If LOAD/STORE have the value in INT sign-magnitude format and SFPU needs it as 2's complement.
    constexpr auto INSTR_MOD_CAST = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // operand B - int32
        TT_SFPLOAD(0 /*lreg*/, INSTR_MOD_LOAD_STORE, ADDR_MOD_7, dst_offset * 64 /*dest_reg_addr */);
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            TTI_SFPCAST(0 /*lreg*/, 2 /*ldest*/, INSTR_MOD_CAST);
            // Required after cast due to a bug in Blackhole RTL.
            TTI_SFPSETSGN(0 /* imm */, 2 /*lreg_c*/, 0 /*ldest*/, 0 /*imod*/);
        }

        // operand A - int32
        TTI_SFPLOAD(1 /*lreg*/, INSTR_MOD_LOAD_STORE, ADDR_MOD_7, 0);
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            TTI_SFPCAST(1 /*lreg*/, 2 /*ldest*/, INSTR_MOD_CAST);
            // Required after cast due to a bug in Blackhole RTL.
            TTI_SFPSETSGN(0 /* imm */, 2 /*lreg_c*/, 1 /*ldest*/, 0 /*imod*/);
        }

        // Set instruction modifier to 6 to get B's 2's complement
        TTI_SFPIADD(0 /*imm*/, 1 /*lreg_c*/, 0 /*lreg_dest*/, 6 /*imod*/);

        // LREG_0 -> dest as int32
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            TTI_SFPCAST(0 /*lreg*/, 1 /*ldest*/, INSTR_MOD_CAST);
            // Required after cast due to a bug in Blackhole RTL.
            TTI_SFPSETSGN(0 /* imm */, 1 /*lreg_c*/, 0 /*ldest*/, 0 /*imod*/);
        }
        TTI_SFPSTORE(0, INSTR_MOD_LOAD_STORE, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

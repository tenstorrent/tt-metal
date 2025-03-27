// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_max_int32_(const int iterations)
{
    // Modifies LOAD/STORE to work with INT32 2's complement, however
    // in Blackhole this has no effect and format remains INT32 sign-magnitude.
    constexpr auto INSTR_MOD_LOAD_STORE = InstrModLoadStore::INT32_2S_COMP;

    // LOAD/STORE have the value in INT sign magnitude format and SFPU needs it as 2's complement.
    constexpr auto INSTR_MOD_CAST = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;

    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(2 /*lreg*/, INSTR_MOD_LOAD_STORE, ADDR_MOD_7, 0 /*dest_reg_addr */);
        TTI_SFPCAST(2 /*lreg*/, 3 /*ldest*/, INSTR_MOD_CAST);
        // Required after cast due to a bug in Blackhole RTL.
        TTI_SFPSETSGN(0 /* imm */, 3 /*lreg_c*/, 2 /*ldest*/, 0 /*imod*/);

        TTI_SFPLOAD(0 /*lreg*/, INSTR_MOD_LOAD_STORE, ADDR_MOD_7, 64 /*dest_reg_addr */);
        TTI_SFPCAST(0 /*lreg*/, 3 /*ldest*/, INSTR_MOD_CAST);
        // Required after cast due to a bug in Blackhole RTL.
        TTI_SFPSETSGN(0 /* imm */, 3 /*lreg_c*/, 0 /*ldest*/, 0 /*imod*/);

        TTI_SFPMOV(0, 0, 1, 0);
        TTI_SFPIADD(0, 2, 1, 2);

        TTI_SFPCAST(0 /*lreg*/, 1 /*ldest*/, INSTR_MOD_CAST);
        // Required after cast due to a bug in Blackhole RTL.
        TTI_SFPSETSGN(0 /* imm */, 1 /*lreg_c*/, 0 /*ldest*/, 0 /*imod*/);
        TTI_SFPSTORE(0, INSTR_MOD_LOAD_STORE, ADDR_MOD_7, 0);

        TTI_SFPENCC(0x003, 0, 0, 10);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

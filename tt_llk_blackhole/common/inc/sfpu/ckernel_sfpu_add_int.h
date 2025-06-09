// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{
namespace
{
constexpr bool is_valid_instruction_mode(InstrModLoadStore mode)
{
    return mode == InstrModLoadStore::INT32_2S_COMP || mode == InstrModLoadStore::INT32 || mode == InstrModLoadStore::LO16;
}
} // anonymous namespace

template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE = INT32_2S_COMP, bool SIGN_MAGNITUDE_FORMAT = false, int ITERATIONS = 8>
inline void _add_int_(const uint dst_offset)
{
    // Operand A is input1 (int32/uint16/uint32)
    // Operand B is input2 (int32/uint16/uint32)
    // Output is int32/uint16/uint32
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // INSTR_MOD_LOAD_STORE = InstrModLoadStore::INT32_2S_COMP modifies LOAD/STORE
    // to do INT32 sign-magnitude to 2's complement conversion, however
    // in Blackhole this has no effect and format remains in original format.

    // If LOAD/STORE have the value in INT sign-magnitude format and SFPU needs it as 2's complement.
    constexpr auto INSTR_MOD_CAST = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // operand A
        TTI_SFPLOAD(0 /*lreg*/, INSTRUCTION_MODE, ADDR_MOD_7, 0 /*dest_reg_addr */);
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            TTI_SFPCAST(0 /*lreg*/, 2 /*ldest*/, INSTR_MOD_CAST);
            // Required after cast due to a bug in Blackhole RTL.
            TTI_SFPSETSGN(0 /* imm */, 2 /*lreg_c*/, 0 /*ldest*/, 0 /*imod*/);
        }

        // operand B
        TT_SFPLOAD(1 /*lreg*/, INSTRUCTION_MODE, ADDR_MOD_7, dst_offset * 64);
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            TTI_SFPCAST(1 /*lreg*/, 2 /*ldest*/, INSTR_MOD_CAST);
            // Required after cast due to a bug in Blackhole RTL.
            TTI_SFPSETSGN(0 /* imm */, 2 /*lreg_c*/, 1 /*ldest*/, 0 /*imod*/);
        }

        TTI_SFPIADD(0 /*imm*/, 1 /*lreg_c*/, 0 /*lreg_dest*/, 4 /*imod*/);

        // LREG_0 -> dest
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            TTI_SFPCAST(0 /*lreg*/, 1 /*ldest*/, INSTR_MOD_CAST);
            // Required after cast due to a bug in Blackhole RTL.
            TTI_SFPSETSGN(0 /* imm */, 1 /*lreg_c*/, 0 /*ldest*/, 0 /*imod*/);
        }
        TTI_SFPSTORE(0, INSTRUCTION_MODE, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

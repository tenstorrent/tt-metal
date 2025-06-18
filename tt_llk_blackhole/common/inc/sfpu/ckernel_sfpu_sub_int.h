// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _sub_int_(const uint dst_offset)
{
    // Operand A is input1 (int32/uint16)
    // Operand B is input2 (int32/uint16)
    // Output is int32/uint16
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // INSTRUCTION_MODE = InstrModLoadStore::INT32_2S_COMP enables LOAD/STORE operations to convert INT32 sign-magnitude to 2's complement.
    // However, in Blackhole, this mode has no effect and the data format remains unchanged.

    // If LOAD/STORE have the value in INT sign-magnitude format and SFPU needs it as 2's complement.
    constexpr auto INSTR_MOD_CAST = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // operand B
        TT_SFPLOAD(p_sfpu::LREG0 /*lreg*/, INSTRUCTION_MODE, ADDR_MOD_7, dst_offset * 64 /*dest_reg_addr */);
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG2, INSTR_MOD_CAST);
        }

        // operand A
        TTI_SFPLOAD(p_sfpu::LREG1 /*lreg*/, INSTRUCTION_MODE, ADDR_MOD_7, 0);
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            apply_sign_magnitude_conversion(p_sfpu::LREG1, p_sfpu::LREG2, INSTR_MOD_CAST);
        }

        // Set instruction modifier to 6 to get B's 2's complement
        TTI_SFPIADD(0 /*imm*/, p_sfpu::LREG1 /*lreg_c*/, p_sfpu::LREG0 /*lreg_dest*/, 6 /*imod*/);

        // LREG_0 -> dest
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG1, INSTR_MOD_CAST);
        }
        TTI_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

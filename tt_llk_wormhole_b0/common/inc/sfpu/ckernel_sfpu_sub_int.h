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
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // Use '12' if Dest is in sign-magnitude format and '4' for 2's complement,
    // because TTI_SFPIADD requires 2's complement format in LREGs
    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : static_cast<std::underlying_type_t<InstrModLoadStore>>(INSTRUCTION_MODE);

// Operand A is input1 (int32/uint16)
// Operand B is input2 (int32/uint16)
// Output is int32/uint16
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // operand A
        TTI_SFPLOAD(p_sfpu::LREG1 /*lreg*/, sfpload_instr_mod, ADDR_MOD_3 /*addr_mode*/, 0 /*dest*/);
        // operand B
        TT_SFPLOAD(p_sfpu::LREG0 /*lreg*/, sfpload_instr_mod, ADDR_MOD_3 /*addr_mode*/, dst_offset * 64 /*dest*/);
        // Use 6 or LO16 as imod to convert operand B to 2's complement
        TTI_SFPIADD(0 /*imm*/, p_sfpu::LREG1 /*lreg*/, p_sfpu::LREG0 /*ldest*/, 6 /*imod*/);
        // LREG_0 -> dest
        TTI_SFPSTORE(p_sfpu::LREG0 /*lreg_ind*/, sfpload_instr_mod, ADDR_MOD_3 /*addr_mode*/, 0 /*dest*/);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

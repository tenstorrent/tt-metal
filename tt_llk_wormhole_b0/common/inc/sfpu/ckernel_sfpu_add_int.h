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
inline void _add_int_(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out)
{
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // For int32, use '12' if Dest is in sign-magnitude format and '4' for 2's complement,
    // because TTI_SFPIADD requires 2's complement format in LREGs
    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : static_cast<std::underlying_type_t<InstrModLoadStore>>(INSTRUCTION_MODE);

    // Operand A is input1 (int32/uint16/uint32)
    // Operand B is input2 (int32/uint16/uint32)
    // Output is int32/uint16/uint32

    // size of each tile in Dest is 64 rows
    constexpr uint dst_tile_size = 64;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // operand A
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        // operand B
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        // LREG_0 -> dest
        TT_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

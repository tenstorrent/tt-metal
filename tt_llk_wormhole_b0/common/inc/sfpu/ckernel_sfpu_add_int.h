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

template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE = INT32, bool SIGN_MAGNITUDE_FORMAT = false, int ITERATIONS = 8>
inline void _add_int_(const uint dst_offset)
{
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // For int32, use '12' if Dest is in sign-magnitude format and '4' for 2's complement,
    // because TTI_SFPIADD requires 2's complement format in LREGs
    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : static_cast<std::underlying_type_t<InstrModLoadStore>>(INSTRUCTION_MODE);

    // Operand A is input1 (int32/uint16/uint32)
    // Operand B is input2 (int32/uint16/uint32)
    // Output is int32/uint16/uint32
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // operand A
        TTI_SFPLOAD(0, sfpload_instr_mod, 3, 0);
        // operand B
        TT_SFPLOAD(1, sfpload_instr_mod, 3, dst_offset * 64);
        TTI_SFPIADD(0, 1, 0, 4);
        // LREG_0 -> dest
        TTI_SFPSTORE(0, sfpload_instr_mod, 3, 0);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

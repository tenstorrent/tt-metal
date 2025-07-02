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

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _calculate_binary_left_shift_(const uint dst_offset)
{
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : static_cast<std::underlying_type_t<InstrModLoadStore>>(INSTRUCTION_MODE);
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr uint dst_tile_size = 64;
        // load
        TTI_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, 0);
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_7, dst_offset * dst_tile_size);
        // if (shift_amount < 0 OR shift_amount >= 32) -> result should be 0
        TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, 1); // 0xFE0 = -32
        TTI_SFPCOMPC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        // shift left
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        // store result
        TTI_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _calculate_binary_right_shift_(const uint dst_offset)
{
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : static_cast<std::underlying_type_t<InstrModLoadStore>>(INSTRUCTION_MODE);
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr uint dst_tile_size = 64;
        // load
        TTI_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, 0);
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_7, dst_offset * dst_tile_size);
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0); // save shift_value for later
        // if (shift_amount < 0 OR shift_amount >= 32) -> result should be 0
        TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LCONST_0); // 0xFE0 = -32
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 6); // take negative of shift_amount to shift right
        // shift right
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        // if shift_value was negative, need to shift in 1's manually
        TTI_SFPSETCC(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);    // only run if shift_value is negative
        TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 2);    // only needed if shift_amount>0
        TTI_SFPIADD(0x020, p_sfpu::LREG1, p_sfpu::LREG2, 5); // take 32-shift_amount (0x020 = 32)
        TTI_SFPNOT(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);   // put all 1's into LREG3
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);     // shift all 1's by 32-shift_amount
        TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);       // OR in the 1's
        TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        // store result
        TTI_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _calculate_logical_right_shift_(const uint dst_offset)
{
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : static_cast<std::underlying_type_t<InstrModLoadStore>>(INSTRUCTION_MODE);

    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr uint dst_tile_size = 64;
        // load
        TTI_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, 0);
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_7, dst_offset * dst_tile_size);
        // if (shift_amount < 0 OR shift_amount >= 32) -> result should be 0
        TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, 1); // 0xFE0 = -32
        TTI_SFPCOMPC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        // shift right
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 6); // take negative of shift_amount to shift right
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        // store result
        TTI_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

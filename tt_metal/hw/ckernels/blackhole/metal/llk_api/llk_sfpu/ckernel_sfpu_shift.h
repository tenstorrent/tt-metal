// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void calculate_binary_left_shift(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    constexpr InstrModLoadStore sfpload_instr_mod =
        SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32_2S_COMP : INSTRUCTION_MODE;
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64 rows
        constexpr std::uint32_t dst_tile_size = 64;
        // load
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_7, dst_index_in1 * dst_tile_size);
        // if (shift_amount < 0 OR shift_amount >= 32) -> result should be 0
        TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, 1);  // 0xFE0 = -32
        TTI_SFPCOMPC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        // shift left
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        // store result
        TT_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void calculate_binary_right_shift(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    constexpr InstrModLoadStore sfpload_instr_mod =
        SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32_2S_COMP : INSTRUCTION_MODE;
    // Pick the DataLayout whose load/store format byte matches the InstrModLoadStore
    // (LO16->U16, INT32/2S_COMP->I32; on Blackhole INT32_2S_COMP is raw int32).
    constexpr sfpi::DataLayout layout =
        (sfpload_instr_mod == InstrModLoadStore::LO16) ? sfpi::DataLayout::U16 : sfpi::DataLayout::I32;
    using vType = std::conditional_t<layout == sfpi::DataLayout::U16, sfpi::vUInt, sfpi::vInt>;

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vType a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<layout>();
        vType s = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<layout>();
        sfpi::vUInt value = sfpi::as<sfpi::vUInt>(a);
        sfpi::vInt shift = sfpi::as<sfpi::vInt>(s);

        // Right shift by `shift` (a negative shift amount shifts left). Out-of-range lanes are
        // fixed up to 0 by the final guard below, so their intermediate result here is don't-care.
        sfpi::vUInt result = sfpi::shft(value, 0 - shift, sfpi::ShiftMode::Logical);

        // Arithmetic shift: when the original value is negative (and the shift is non-zero) the
        // vacated high bits must be filled with 1's rather than 0's.
        v_if(sfpi::as<sfpi::vInt>(value) < 0 && shift != 0) {
            sfpi::vUInt high_ones = sfpi::shft(~sfpi::vUInt(0), 32 - shift, sfpi::ShiftMode::Logical);
            result = result | high_ones;
        }
        v_endif;

        // Out-of-range shift amounts (shift < 0 or shift >= 32) produce 0, matching the sibling
        // kernels and the "shift_amount < 0 OR >= 32 -> 0" contract.
        v_if(shift < 0 || shift >= 32) { result = 0; }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<layout>() = sfpi::as<vType>(result);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void calculate_logical_right_shift(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    constexpr InstrModLoadStore sfpload_instr_mod =
        SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32_2S_COMP : INSTRUCTION_MODE;

    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64 rows
        constexpr std::uint32_t dst_tile_size = 64;
        // load
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_7, dst_index_in1 * dst_tile_size);
        // if (shift_amount < 0 OR shift_amount >= 32) -> result should be 0
        TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, 1);  // 0xFE0 = -32
        TTI_SFPCOMPC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        // shift right
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 6);  // take negative of shift_amount to shift right
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        // store result
        TT_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Map the SFP load/store instruction mode to the sfpi DataLayout whose load/store format byte
// matches it (see ckernel_sfpu_rsub_int32.h):
//   INT32 (4)          -> I32  (sign-mag<->2's-comp conversion on load/store)
//   LO16 (6)           -> U16
//   INT32_2S_COMP (12) -> SM32 (raw)
template <InstrModLoadStore INSTRUCTION_MODE>
inline constexpr sfpi::DataLayout shift_layout() {
    return (INSTRUCTION_MODE == InstrModLoadStore::LO16)            ? sfpi::DataLayout::U16
           : (INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP) ? sfpi::DataLayout::SM32
                                                                    : sfpi::DataLayout::I32;
}

// The in-register value type that the layout can load/store: U16 holds unsigned lanes,
// I32/SM32 hold signed (2's-complement) lanes.
template <sfpi::DataLayout LAYOUT>
using shift_vtype = std::conditional_t<LAYOUT == sfpi::DataLayout::U16, sfpi::vUInt, sfpi::vInt>;

constexpr std::uint32_t dst_tile_size = 32;

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void calculate_binary_left_shift(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    constexpr InstrModLoadStore sfpload_instr_mod =
        SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32_2S_COMP : INSTRUCTION_MODE;
    constexpr sfpi::DataLayout layout = shift_layout<sfpload_instr_mod>();
    using vType = shift_vtype<layout>;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vType a = sfpi::dst_reg[dst_index_in0 * dst_tile_size].mode<layout>();
        vType s = sfpi::dst_reg[dst_index_in1 * dst_tile_size].mode<layout>();
        sfpi::vUInt value = sfpi::as<sfpi::vUInt>(a);
        sfpi::vInt shift = sfpi::as<sfpi::vInt>(s);

        // A positive shift amount shifts left (logical).
        sfpi::vUInt result = sfpi::shft(value, shift, sfpi::ShiftMode::Logical);

        // Out-of-range shift amounts (shift < 0 or shift >= 32) produce 0.
        v_if(shift < 0 || shift >= 32) { result = 0; }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size].mode<layout>() = sfpi::as<vType>(result);
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
    constexpr sfpi::DataLayout layout = shift_layout<sfpload_instr_mod>();
    using vType = shift_vtype<layout>;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vType a = sfpi::dst_reg[dst_index_in0 * dst_tile_size].mode<layout>();
        vType s = sfpi::dst_reg[dst_index_in1 * dst_tile_size].mode<layout>();
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
        // kernels and the original "shift_amount < 0 OR >= 32 -> 0" contract.
        v_if(shift < 0 || shift >= 32) { result = 0; }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size].mode<layout>() = sfpi::as<vType>(result);
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
    constexpr sfpi::DataLayout layout = shift_layout<sfpload_instr_mod>();
    using vType = shift_vtype<layout>;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vType a = sfpi::dst_reg[dst_index_in0 * dst_tile_size].mode<layout>();
        vType s = sfpi::dst_reg[dst_index_in1 * dst_tile_size].mode<layout>();
        sfpi::vUInt value = sfpi::as<sfpi::vUInt>(a);
        sfpi::vInt shift = sfpi::as<sfpi::vInt>(s);

        // Out-of-range shift amounts (shift < 0 or shift >= 32) produce 0. Zeroing the value up
        // front (rather than the result afterwards) means the single shift below still yields 0 for
        // those lanes, and lets the compiler negate `shift` in place instead of keeping a live copy.
        v_if(shift < 0 || shift >= 32) { value = 0; }
        v_endif;

        // Right shift by `shift` (a negative shift amount shifts left).
        sfpi::vUInt result = sfpi::shft(value, 0 - shift, sfpi::ShiftMode::Logical);

        sfpi::dst_reg[dst_index_out * dst_tile_size].mode<layout>() = sfpi::as<vType>(result);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

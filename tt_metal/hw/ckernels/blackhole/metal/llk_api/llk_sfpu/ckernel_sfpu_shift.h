// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

// The sfpi DataLayout whose load/store format byte matches the InstrModLoadStore the caller asked
// for: LO16 (6) is sfpi's U16, INT32 (4) is sfpi's I32. On Blackhole INT32_2S_COMP loads the same
// raw bits as INT32 -- the sign-magnitude conversion its name implies has no effect there (see
// _add_int_) -- so it maps to I32 as well. Mapping it to SM32 instead would make sfpi wrap every
// load and store in a sign-magnitude conversion.
template <InstrModLoadStore INSTRUCTION_MODE>
inline constexpr sfpi::DataLayout shift_layout() {
    return (INSTRUCTION_MODE == InstrModLoadStore::LO16) ? sfpi::DataLayout::U16 : sfpi::DataLayout::I32;
}

// The in-register value type the layout can load and store: U16 holds unsigned lanes, I32 holds
// signed (2's-complement) lanes.
template <sfpi::DataLayout LAYOUT>
using shift_vtype = std::conditional_t<LAYOUT == sfpi::DataLayout::U16, sfpi::vUInt, sfpi::vInt>;

// sfpi walks Dest in units of SFP_DESTREG_STRIDE, so a tile is 32 rows here rather than the 64
// rows a raw TT_SFPLOAD dest-row immediate counts.
constexpr std::uint32_t dst_tile_size_sfpi = 32;

// A shift amount is in range exactly when it has no bit set outside [0, 32), which covers the
// negative amounts too. Testing the mask costs one SFPAND against a loop-invariant register,
// where `shift < 0 || shift >= 32` costs an SFPSETCC/SFPIADD/SFPCOMPC triple.
constexpr int shift_out_of_range_mask = ~31;

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void calculate_binary_left_shift(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    constexpr InstrModLoadStore sfpload_instr_mod =
        SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32_2S_COMP : INSTRUCTION_MODE;
    constexpr sfpi::DataLayout layout = shift_layout<sfpload_instr_mod>();
    using vType = shift_vtype<layout>;

    const sfpi::vInt out_of_range = shift_out_of_range_mask;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vType a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<layout>();
        vType s = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<layout>();
        sfpi::vInt shift = sfpi::as<sfpi::vInt>(s);

        // A positive shift amount shifts left.
        sfpi::vUInt result = sfpi::shft(sfpi::as<sfpi::vUInt>(a), shift, sfpi::ShiftMode::Logical);

        // A shift amount outside [0, 32) produces 0.
        v_if((shift & out_of_range) != 0) { result = 0; }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<layout>() = sfpi::as<vType>(result);
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

    const sfpi::vInt out_of_range = shift_out_of_range_mask;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vType a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<layout>();
        vType s = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<layout>();
        sfpi::vInt value = sfpi::as<sfpi::vInt>(a);
        sfpi::vInt shift = sfpi::as<sfpi::vInt>(s);

        // A shift amount outside [0, 32) produces 0, not the -1 an unbounded arithmetic shift of a
        // negative operand would give. Zeroing the operand rather than the result keeps the single
        // shift below correct for those lanes and lets the negation happen in place.
        v_if((shift & out_of_range) != 0) { value = 0; }
        v_endif;

        // Blackhole's SFPSHFT fills the vacated high bits from the sign bit in arithmetic mode, so
        // a negative operand needs no hand-rolled mask of 1's. A negative amount shifts right.
        sfpi::vInt result = sfpi::shft(value, -shift, sfpi::ShiftMode::Arithmetic);

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
    constexpr sfpi::DataLayout layout = shift_layout<sfpload_instr_mod>();
    using vType = shift_vtype<layout>;

    const sfpi::vInt out_of_range = shift_out_of_range_mask;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vType a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<layout>();
        vType s = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<layout>();
        sfpi::vUInt value = sfpi::as<sfpi::vUInt>(a);
        sfpi::vInt shift = sfpi::as<sfpi::vInt>(s);

        // A shift amount outside [0, 32) produces 0. Zeroing the operand rather than the result
        // keeps the single shift below correct for those lanes and lets the negation happen in
        // place.
        v_if((shift & out_of_range) != 0) { value = 0; }
        v_endif;

        // A negative amount shifts right.
        sfpi::vUInt result = sfpi::shft(value, -shift, sfpi::ShiftMode::Logical);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<layout>() = sfpi::as<vType>(result);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void calculate_clamped_logical_right_shift(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        INSTRUCTION_MODE == InstrModLoadStore::INT32,
        "Clamped logical right shift is only used for UInt32 operands in INT32 mode.");

    const sfpi::vInt out_of_range = shift_out_of_range_mask;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vInt shift = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vUInt value = sfpi::as<sfpi::vUInt>(a);

        v_if((shift & out_of_range) != 0) { shift = 31; }
        v_endif;

        sfpi::vUInt result = sfpi::shft(value, -shift, sfpi::ShiftMode::Logical);
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = sfpi::as<sfpi::vInt>(result);
        sfpi::dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// BinaryShift<APPROXIMATION_MODE, SHIFT_OP, INSTRUCTION_MODE, DST_SYNC, DST_ACCUM, SIGN_MAGNITUDE_FORMAT, ITERATIONS>
//   SHIFT_OP selects calculate_binary_left_shift / calculate_binary_right_shift / calculate_logical_right_shift /
//   calculate_clamped_logical_right_shift.
//   init() is the shared SFPU init only.
//   Backs binary_left_shift_tile, binary_right_shift_tile, binary_logical_right_shift_tile,
//   binary_shift_tile_init (api/compute/binary_shift.h).
// ---------------------------------------------------------------------------------------------------
enum class BinaryShiftOp : std::uint8_t {
    LEFT = 0,
    RIGHT = 1,
    LOGICAL_RIGHT = 2,
    CLAMPED_LOGICAL_RIGHT = 3,
};

template <
    bool APPROXIMATION_MODE,
    BinaryShiftOp SHIFT_OP,
    InstrModLoadStore INSTRUCTION_MODE,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    bool SIGN_MAGNITUDE_FORMAT = false,
    int ITERATIONS = 8>
struct BinaryShift : SfpuBinaryOp<
                         BinaryShift<
                             APPROXIMATION_MODE,
                             SHIFT_OP,
                             INSTRUCTION_MODE,
                             DST_SYNC,
                             DST_ACCUM,
                             SIGN_MAGNITUDE_FORMAT,
                             ITERATIONS>,
                         DST_SYNC,
                         DST_ACCUM> {
    static void kernel(std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_out) {
        if constexpr (SHIFT_OP == BinaryShiftOp::LEFT) {
            calculate_binary_left_shift<APPROXIMATION_MODE, ITERATIONS, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT>(
                dst_index_in0, dst_index_in1, dst_index_out);
        } else if constexpr (SHIFT_OP == BinaryShiftOp::RIGHT) {
            calculate_binary_right_shift<APPROXIMATION_MODE, ITERATIONS, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT>(
                dst_index_in0, dst_index_in1, dst_index_out);
        } else if constexpr (SHIFT_OP == BinaryShiftOp::LOGICAL_RIGHT) {
            calculate_logical_right_shift<APPROXIMATION_MODE, ITERATIONS, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT>(
                dst_index_in0, dst_index_in1, dst_index_out);
        } else {
            calculate_clamped_logical_right_shift<
                APPROXIMATION_MODE,
                ITERATIONS,
                INSTRUCTION_MODE,
                SIGN_MAGNITUDE_FORMAT>(dst_index_in0, dst_index_in1, dst_index_out);
        }
    }
};
}  // namespace sfpu
}  // namespace ckernel

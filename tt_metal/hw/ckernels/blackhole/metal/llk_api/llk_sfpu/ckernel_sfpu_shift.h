// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpi.h"

// Blackhole binary integer shift kernels, migrated from raw-TTI microcode to SFPI.
//
// The previous hand-written microcode operated on raw LREG bit patterns and hand-rolled
// both the sign extension (arithmetic right shift) and the out-of-range predication. That
// only produced correct results for a sign-magnitude LREG representation and diverged from
// the two's-complement golden for negative operands under INT32_2S_COMP.
//
// Dst physically stores int32 in sign-magnitude, so operands must be read through sfpi's
// `DataLayout::SM32` mode: on load it applies smag_to_int() to yield true two's-complement
// `vInt` lanes, and on store it applies int_to_smag() to convert back. (Plain `DataLayout::I32`
// reads/writes the raw sign-magnitude bits and is only correct for non-negative values.)
// Blackhole's native arithmetic right shift (`vInt >> vUInt`) then sign-extends without any
// manual bit twiddling. See docs/SFPU_INT32_SHIFT.md for the full analysis.
//
// Semantics (per element, matching BinarySFPUGolden):
//   shift < 0 OR shift >= 32          -> 0   (all three ops)
//   right shift (32-bit)              -> arithmetic (sign-extending)
//   logical right shift               -> unsigned (zero-fill)
//   left shift                        -> plain left shift
// The shift amount is a per-lane second operand, so the out-of-range mask is evaluated
// per lane. INT32_MIN cannot round-trip through the sign-magnitude Dst and is unsupported.
//
// The template signature is kept identical to the raw-TTI version so the compute-API and
// test call sites are unchanged; INSTRUCTION_MODE only selects the element width
// (LO16 -> 16-bit lanes, otherwise 32-bit). SIGN_MAGNITUDE_FORMAT is no longer needed
// because sfpi's typed layouts handle the Dst representation.

namespace ckernel {
namespace sfpu {

// sfpi's dst_reg[] indexes in sfpi row units (32 rows/tile), unlike the raw TT_SFPLOAD
// immediate which used 64.
constexpr std::uint32_t dst_tile_size_sfpi_shift = 32;

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void calculate_binary_left_shift(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");
    constexpr bool is_16bit = (INSTRUCTION_MODE == InstrModLoadStore::LO16);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (is_16bit) {
            sfpi::vUInt v = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::U16>();
            sfpi::vUInt amt = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::U16>();
            sfpi::vUInt res = v << amt;
            v_if(amt >= sfpi::vUInt(32u)) { res = sfpi::vUInt(0u); }
            v_endif;
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::U16>() = res;
        } else {
            sfpi::vInt v = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::SM32>();
            sfpi::vInt amt = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::SM32>();
            sfpi::vInt res = v << sfpi::as<sfpi::vUInt>(amt);
            // shift amount outside [0, 31] -> 0
            v_if(amt < 0) { res = sfpi::vInt(0); }
            v_endif;
            v_if(amt >= 32) { res = sfpi::vInt(0); }
            v_endif;
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::SM32>() = res;
        }
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void calculate_binary_right_shift(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");
    constexpr bool is_16bit = (INSTRUCTION_MODE == InstrModLoadStore::LO16);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (is_16bit) {
            sfpi::vUInt v = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::U16>();
            sfpi::vUInt amt = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::U16>();
            sfpi::vUInt res = v >> amt;
            v_if(amt >= sfpi::vUInt(32u)) { res = sfpi::vUInt(0u); }
            v_endif;
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::U16>() = res;
        } else {
            sfpi::vInt v = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::SM32>();
            sfpi::vInt amt = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::SM32>();
            // Native Blackhole arithmetic (sign-extending) right shift on true 2's-comp lanes.
            sfpi::vInt res = v >> sfpi::as<sfpi::vUInt>(amt);
            // shift amount outside [0, 31] -> 0
            v_if(amt < 0) { res = sfpi::vInt(0); }
            v_endif;
            v_if(amt >= 32) { res = sfpi::vInt(0); }
            v_endif;
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::SM32>() = res;
        }
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void calculate_logical_right_shift(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");
    constexpr bool is_16bit = (INSTRUCTION_MODE == InstrModLoadStore::LO16);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (is_16bit) {
            sfpi::vUInt v = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::U16>();
            sfpi::vUInt amt = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::U16>();
            sfpi::vUInt res = v >> amt;
            v_if(amt >= sfpi::vUInt(32u)) { res = sfpi::vUInt(0u); }
            v_endif;
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::U16>() = res;
        } else {
            sfpi::vInt v = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::SM32>();
            sfpi::vInt amt = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::SM32>();
            // Unsigned (logical, zero-fill) right shift regardless of the sign bit.
            sfpi::vUInt res = sfpi::as<sfpi::vUInt>(v) >> sfpi::as<sfpi::vUInt>(amt);
            // shift amount outside [0, 31] -> 0
            v_if(amt < 0) { res = sfpi::vUInt(0u); }
            v_endif;
            v_if(amt >= 32) { res = sfpi::vUInt(0u); }
            v_endif;
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi_shift].mode<sfpi::DataLayout::SM32>() =
                sfpi::as<sfpi::vInt>(res);
        }
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

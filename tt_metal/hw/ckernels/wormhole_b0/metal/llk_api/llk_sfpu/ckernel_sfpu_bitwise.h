// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

enum class UnaryBitwiseOp : std::uint8_t {
    AND = 0,
    OR = 1,
    XOR = 2,
};

template <UnaryBitwiseOp BITWISE_OP, typename Vec>
inline Vec compute_unary_bitwise(Vec v, Vec scalar) {
    if constexpr (BITWISE_OP == UnaryBitwiseOp::AND) {
        return v & scalar;
    } else if constexpr (BITWISE_OP == UnaryBitwiseOp::OR) {
        return v | scalar;
    } else {
        static_assert(BITWISE_OP == UnaryBitwiseOp::XOR, "Unsupported bitwise op");
        return v ^ scalar;
    }
}

// The store in calculate_sfpu_unary_bitwise advances the destination by one SFPU row
// (dest.incr = 2, what dst_reg++ does) through ADDR_MOD_6, so each row saves a TTINCRWC.
// The unary SFPU init runs this per-op init rather than the ADDR_MOD_6 whitelist in
// eltwise_unary_sfpu_configure_addrmod, so the address mod is programmed here (as square_init
// and unary_max_min_init do).
inline void bitwise_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void bitwise_and_init() { bitwise_init(); }

inline void bitwise_or_init() { bitwise_init(); }

inline void bitwise_xor_init() { bitwise_init(); }

template <
    bool APPROXIMATION_MODE,
    UnaryBitwiseOp BITWISE_OP,
    DataFormat DATA_FORMAT = DataFormat::Int32,
    int ITERATIONS = 8>
inline void calculate_sfpu_unary_bitwise(const uint value) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for bitwise operation. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (DATA_FORMAT == DataFormat::UInt16) {
        // Bitwise op is per-lane (v & scalar), so `scalar` must be a vector broadcast of `value`
        // whose type matches the loaded `v` (vUInt for the U16 layout).
        const sfpi::vUInt scalar = static_cast<unsigned>(value);
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vUInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>();
            // ADDR_MOD_6 (bitwise_init) advances the destination on the store, in place of a
            // separate dst_reg++. Wormhole's SFPU load and store pick ADDR_MOD_4..7 with a
            // two-bit field, so ADDR_MOD_6 is selected as 2, as in square and unary max/min.
            sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>(ADDR_MOD_2) = compute_unary_bitwise<BITWISE_OP>(v, scalar);
        }
    } else {
        // I32 layout loads vInt, so `scalar` must be vInt to match `v`.
        const sfpi::vInt scalar = static_cast<int>(value);
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            // See the UInt16 branch: the store advances the destination through ADDR_MOD_6.
            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>(ADDR_MOD_2) = compute_unary_bitwise<BITWISE_OP>(v, scalar);
        }
    }
}
}  // namespace sfpu
}  // namespace ckernel

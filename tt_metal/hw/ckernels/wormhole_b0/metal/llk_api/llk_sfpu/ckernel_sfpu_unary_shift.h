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

inline void left_shift_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

// Left shift by an immediate scalar amount. If shift amount is >= 32, the result is 0.
template <bool APPROXIMATION_MODE, DataFormat DATA_FORMAT = DataFormat::Int32, int ITERATIONS = 8>
inline void calculate_left_shift(const uint shift_amt) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for shift operation. Supported data formats are: Int32, UInt32, UInt16");
    const bool out_of_range = shift_amt >= 32;
    // SFPI overloads both `vInt << unsigned` and `vUInt << unsigned`, so the shift amount's type is
    // independent of the element type being shifted. Cast to a 32-bit `unsigned` so shift is chosen exactly.
    const unsigned amt = static_cast<unsigned>(shift_amt);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (DATA_FORMAT == DataFormat::UInt16) {
            sfpi::vUInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>();
            sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>() = out_of_range ? sfpi::vUInt(0u) : (v << amt);
        } else {
            sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = out_of_range ? sfpi::vInt(0) : (v << amt);
        }
        sfpi::dst_reg++;
    }
}

inline void right_shift_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

// Arithmetic right shift by an immediate scalar amount.
// A shift amount >= 32 saturates to the sign (non-negative -> 0, negative -> -1).
template <bool APPROXIMATION_MODE, DataFormat DATA_FORMAT = DataFormat::Int32, int ITERATIONS = 8>
inline void calculate_right_shift(const uint shift_amt) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for shift operation. Supported data formats are: Int32, UInt32, UInt16");
    // SFPI overloads both `vInt << unsigned` and `vUInt << unsigned`, so the shift amount's type is
    // independent of the element type being shifted. Cast to a 32-bit `unsigned` so shift is chosen exactly.
    const unsigned eff = (shift_amt >= 32) ? 31u : static_cast<unsigned>(shift_amt);
    const unsigned sign_mask = (eff > 0) ? (~0u << (32 - eff)) : 0u;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (DATA_FORMAT == DataFormat::UInt16) {
            sfpi::vUInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>();
            sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>() = v >> eff;
        } else {
            sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            sfpi::vUInt res = sfpi::as<sfpi::vUInt>(v) >> eff;
            v_if(v < 0) { res = res | sign_mask; }
            v_endif;
            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = sfpi::as<sfpi::vInt>(res);
        }
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

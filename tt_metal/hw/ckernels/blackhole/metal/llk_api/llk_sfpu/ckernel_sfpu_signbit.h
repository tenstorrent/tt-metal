// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// signbit(x) = 1.0f if x < 0, 0.0f otherwise (including -0.0f → 1.0f)
// Branchless implementation using arithmetic right shift
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit() {
    constexpr uint32_t one_float_bits = 0x3F800000;  // IEEE 754: 1.0f

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];

        // Arithmetic right shift by 31 produces:
        //   0xFFFFFFFF for negative (sign bit = 1)
        //   0x00000000 for positive (sign bit = 0)
        sfpi::vInt bits = sfpi::reinterpret<sfpi::vInt>(val);
        sfpi::vInt mask = bits >> 31;

        // Mask selects 1.0f for negative, 0.0f for positive
        sfpi::vInt result_bits = mask & one_float_bits;

        sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(result_bits);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit_int32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_7, 0);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

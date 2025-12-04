// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

// Arithmetic right shift for signed integers
// Implements sign-extending right shift using logical shift + bit flipping
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_right_shift(const uint shift_amt) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt input = sfpi::dst_reg[0];
        sfpi::vUInt val = sfpi::reinterpret<sfpi::vUInt>(input);

        // For negative numbers: flip bits, shift, flip back
        // This converts logical shift to arithmetic shift
        v_if(input < 0) { val = ~val; }
        v_endif;

        sfpi::vInt res = sfpi::reinterpret<sfpi::vInt>(val >> shift_amt);

        v_if(input < 0) { res = ~res; }
        v_endif;

        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// Implementation uses the SFPU's built-in tanh function for maximum accuracy.
// The cap parameter is passed as a packed float in FP16_B format.

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(uint32_t param0) {
    // Convert packed parameter back to float
    sfpi::vFloat cap = sfpi::s2vFloat16b(param0);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];  // Load input value

        // Compute x / cap
        sfpi::vFloat scaled_x = x / cap;

        // Compute tanh(x / cap) using SFPU tanh function
        sfpi::vFloat tanh_result = sfpi::tanh(scaled_x);

        // Compute result = cap * tanh(x / cap)
        sfpi::vFloat result = cap * tanh_result;

        sfpi::dst_reg[0] = result;  // Store result
        sfpi::dst_reg++;            // Advance to next DEST row
    }
}

}  // namespace sfpu
}  // namespace ckernel

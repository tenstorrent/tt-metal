// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// RReLU (Randomized Leaky ReLU):
//   output = x           if x >= 0
//   output = slope * x   if x < 0
//
// In evaluation mode, slope = (lower + upper) / 2, precomputed on host.
// The slope is passed as a uint32_t containing the bfloat16 bit pattern.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(std::uint32_t slope_u32) {
    sfpi::vFloat slope = sfpi::s2vFloat16b(slope_u32);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];

        v_if(val < 0.0f) { val = val * slope; }
        v_endif;

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

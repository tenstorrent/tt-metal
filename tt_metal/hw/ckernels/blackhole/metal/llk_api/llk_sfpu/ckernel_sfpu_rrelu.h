// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu {

// RReLU (Randomized Leaky ReLU) in evaluation mode:
//   rrelu(x) = x          if x >= 0
//   rrelu(x) = slope * x  if x < 0
//
// where slope = (lower + upper) / 2, pre-computed on the host.
// param0: slope packed as FP16_B (bfloat16 bits in a uint32_t).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint32_t param0) {
    sfpi::vFloat slope = sfpi::s2vFloat16b(param0);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = x;

        v_if(x < 0.0f) { result = x * slope; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

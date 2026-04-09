// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel {
namespace sfpu {

// RReLU (Randomized Leaky ReLU) - Evaluation mode:
//   output = x           if x >= 0
//   output = slope * x   if x < 0
//   where slope = (lower + upper) / 2 (pre-computed on host)
//
// The slope parameter is passed as a uint32_t containing the
// bit-cast representation of the float slope value.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(std::uint32_t slope_param) {
    // Convert the uint32_t parameter back to vFloat
    sfpi::vFloat slope = Converter::as_float(slope_param);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // For negative values, multiply by slope
        v_if(v < 0.0f) { v = v * slope; }
        v_endif;

        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

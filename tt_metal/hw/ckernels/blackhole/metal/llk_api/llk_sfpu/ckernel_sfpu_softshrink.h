// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// softshrink(x, lambda) =
//   x - lambda   if x > lambda
//   x + lambda   if x < -lambda
//   0            otherwise
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softshrink(std::uint32_t param0) {
    // param0 = lambda as IEEE 754 float bits (bitcast uint32_t)
    sfpi::vFloat lambda_val = Converter::as_float(param0);
    sfpi::vFloat neg_lambda = -lambda_val;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = 0.0f;

        v_if(val > lambda_val) { result = val - lambda_val; }
        v_endif;

        v_if(val < neg_lambda) { result = val + lambda_val; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

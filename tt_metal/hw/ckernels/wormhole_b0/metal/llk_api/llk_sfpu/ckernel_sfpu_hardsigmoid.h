// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// hardsigmoid(x) = max(0, min(1, x/6 + 0.5))
// Piecewise linear:
//   x <= -3  =>  0
//   x >= 3   =>  1
//   else     =>  x * (1/6) + 0.5
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardsigmoid() {
    constexpr float one_sixth = 1.0f / 6.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = x * one_sixth + 0.5f;

        // Clamp to [0, 1]
        v_if(result < 0.0f) { result = 0.0f; }
        v_endif;
        v_if(result > sfpi::vConst1) { result = sfpi::vConst1; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

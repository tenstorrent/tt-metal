// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// hardswish(x) = x * min(max(x + 3, 0), 6) / 6
//              = x * hardsigmoid(x)
//              = x * clamp(x/6 + 0.5, 0, 1)
// Piecewise:
//   x <= -3  =>  0
//   x >= 3   =>  x
//   else     =>  x * (x/6 + 0.5)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardswish() {
    constexpr float one_sixth = 1.0f / 6.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat hsigmoid = x * one_sixth + 0.5f;

        // Clamp hardsigmoid to [0, 1]
        v_if(hsigmoid < 0.0f) { hsigmoid = 0.0f; }
        v_endif;
        v_if(hsigmoid > sfpi::vConst1) { hsigmoid = sfpi::vConst1; }
        v_endif;

        sfpi::dst_reg[0] = x * hsigmoid;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

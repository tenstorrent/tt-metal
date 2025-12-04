// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

// heaviside(x, s) = 0 if x < 0, s if x == 0, 1 if x > 0
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_heaviside(uint value) {
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        v_if(v < 0.0f) { v = sfpi::vConst0; }
        v_elseif(v > 0.0f) { v = sfpi::vConst1; }
        v_else { v = s; }
        v_endif;

        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

// PReLU(x, alpha) = x if x >= 0, alpha * x if x < 0
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_prelu(const uint value) {
    sfpi::vFloat alpha = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        v_if(x < sfpi::vConst0) { x *= alpha; }
        v_endif;
        sfpi::dst_reg[0] = x;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

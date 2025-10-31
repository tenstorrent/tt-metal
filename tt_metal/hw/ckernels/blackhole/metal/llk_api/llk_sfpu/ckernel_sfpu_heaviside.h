// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_heaviside(uint value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        v_if(v < 0.0f) { v = 0.0f; }
        v_elseif(v > 0.0f) { v = 1.0f; }
        v_else { v = s; }
        v_endif;

        sfpi::dst_reg[0] = v;

        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

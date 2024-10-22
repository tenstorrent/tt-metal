// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void rand_init(uint seed) {
    init_prng_seed(seed);
}

template <bool APPROXIMATION_MODE>
inline void rand(float from, float to) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        // Generate random float
        TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8);
        // Unset sign bit and Set exponent to 127 to ensure the float is within the range [1, 2).
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPSETEXP(127, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        // TTI_SFPADDI(0xbfa7, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(0, 4, 3, 0);

        // -1 to ensure the float is within the range [0, 1).
        vFloat rand_floats = dst_reg[0];
        // rand_floats -= vConst1;

        vFloat v_from = from;
        vFloat v_to = to;
        rand_floats = (rand_floats - vConst1) * (v_to - v_from) + v_from;
        dst_reg[0] = rand_floats;

        dst_reg++;
    }
}
}  // namespace ckernel::sfpu

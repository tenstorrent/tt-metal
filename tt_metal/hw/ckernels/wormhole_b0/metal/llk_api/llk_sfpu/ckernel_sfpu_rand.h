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
inline void rand() {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8);
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPSETEXP(127, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        // TTI_SFPADDI(0xbfa7, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(0, 4, 3, 0);

        vFloat from = 2.1;
        vFloat to = 5;

        vFloat rand_floats = dst_reg[0];
        rand_floats = (rand_floats - 1) * (to - from) + from;
        dst_reg[0] = rand_floats;

        dst_reg++;
    }
}
}  // namespace ckernel::sfpu

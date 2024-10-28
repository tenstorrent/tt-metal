// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void rand_init(uint32_t seed) {
    init_prng_seed(seed);
}

template <bool APPROXIMATION_MODE>
inline void rand(uint32_t from, uint32_t scale) {
    // Load scale param to lreg1
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);

    // Load from param to lreg2
    TT_SFPLOADI(p_sfpu::LREG2, 10, from & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, from >> 16);

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        // Generate random float
        TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8);

        // Unset sign bit and Set exponent to 127 to ensure the float is within the range [1, 2).
        // lreg0.sign = 0
        // lreg0 = {sign: 0, exponent: 127, mantissa: lreg0.mantissa}
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPSETEXP(127, p_sfpu::LREG0, p_sfpu::LREG0, 1);

        // -1 to ensure the float is within the range [0, 1).
        // lreg0 = lreg0 - 1
        TTI_SFPADDI(0xbf80 /*-1*/, p_sfpu::LREG0, 0);
        TTI_SFPNOP;

        // Scale the float from [0, 1) to [from, from + scale)
        // lreg0 = lreg0 * scale + from
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 1);
        TTI_SFPNOP;

        TTI_SFPSTORE(0, 3, 3, 0);
        dst_reg++;
    }
}
}  // namespace ckernel::sfpu

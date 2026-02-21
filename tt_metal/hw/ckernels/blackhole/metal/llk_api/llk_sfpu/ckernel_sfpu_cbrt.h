// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"

namespace ckernel::sfpu {

// This is a modified version of "Fast Calculation of Cube and Inverse Cube
// Roots Using a Magic Constant and Its Implementation on Microcontrollers" by
// Moroz et al. <https://doi.org/10.3390/en14041058>

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cube_root() {
    sfpi::vFloat third_128 = 0x1.555556p-9f;
    // Magic constant from paper, with MSB set to 1.
    sfpi::vInt magic = 0xd48c2b4b;
    sfpi::vFloat rounding_bias = 8388608.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat x = sfpi::abs(a);

        sfpi::vInt i = sfpi::reinterpret<sfpi::vInt>(x);

        // The original paper wants i = `0x54a223b4 - i/3`.
        // Since computing `i/3` is expensive, we note the following:
        // 1. x is positive, hence the MSB of i is always zero and i<2^31.
        // 2. dividing by 3 loses another bit, so the 2 high bits of i/3 are zero and i/3<2^30.
        // We would like to end up with a value < 2^23 so that we can use the
        // rounding bias trick; adding 2^23 in floating point shifts the value
        // into the mantissa bits.  Since i/3<2^30, we simply need to divide by
        // 2^7 to obtain a value < 2^23.
        //
        // The calculation below does:
        // 1. f = (float)i; this is inexact for values larger than 2^24.
        // 2. Use a single SFPMAD to compute (f/3.0)/128.0, which is guaranteed to be smaller than 2^23,
        //    and add 2^23, shifting result into mantissa bits (rounding to nearest even).
        // 3. Now we extract the mantissa bits, and left-shift by 7 to undo division by 128.0.
        // This gives us approximately `i/3` but with low 7 bits all zero.

        sfpi::vFloat f = sfpi::int32_to_float(i, 0);

        // Workaround for SFPI's insistence on generating SFPADDI+SFPMUL instead of SFPLOADI+SFPMAD here.
        f.get() = __builtin_rvtt_sfpmad(f.get(), third_128.get(), rounding_bias.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

        // f has exponent 23; i<<7 will have two high bits 10; use modified
        // magic constant from paper with MSB=1 to cancel the MSB=1 of i<<7.
        i = sfpi::reinterpret<sfpi::vInt>(f);
        i = magic - (i << 7);

        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(i);

        if constexpr (is_fp32_dest_acc_en) {
            sfpi::vFloat c = (x * y) * (y * y);
            y = y * (c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0);

            sfpi::vFloat d = x * (y * y);
            c = d * y + sfpi::vConstNeg1;
            sfpi::vFloat third = sfpi::addexp(third_128, 7);
            sfpi::vFloat t = c * third + sfpi::vConstNeg1;
            d = sfpi::setsgn(d, a);
            y = d * (t * t);

            sfpi::dst_reg[0] = y;
        } else {
            sfpi::vFloat d = x * (y * y);
            sfpi::vFloat c = d * y;
            sfpi::vFloat t = c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0;
            d = sfpi::setsgn(d, a);
            y = d * (t * t);

            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
        }
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void cube_root_init() {
    sfpi::vConstFloatPrgm0 = 0x1.c09806p0f;
    sfpi::vConstFloatPrgm1 = -0x1.403e6cp0f;
    sfpi::vConstFloatPrgm2 = 0x1.04cdb2p-1f;
}

}  // namespace ckernel::sfpu

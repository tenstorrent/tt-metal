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
    sfpi::vFloat negative_third_256 = -0x1.555556p-10f;

    // Magic constant 0x548c2b4b / 256 + 2^23
    sfpi::vFloat magic = 1418472267.0f / 256.0f + 8388608.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat x = sfpi::abs(a);

        // Paper wants i = 0x548c2b4b - i/3.
        // Due to lack of integer division and lack of fp32 to u32 cast, we
        // compute this using two instructions: SFPMAD and SFPSHFT.
        //
        // First, we compute (0x548c2b4b - i/3) in fp32, but we also need to
        // add 2^23 to shift the result into the mantissa bits for extraction
        // as integer.  This only works if (0x548c2b4b - i/3)*k < 2^23, so we
        // divide everything by 2^8.
        //
        // f = (0x548c2b4b - i * 1.0/3.0) / 256.0 + 2^23
        //   = (0x548c2b4b/256.0 - i * 1.0/3.0/256.0) + 2^23

        sfpi::vFloat f = sfpi::int32_to_float(sfpi::reinterpret<sfpi::vInt>(x), 0);

        f = f * negative_third_256 + magic;

        // Now, left-shift by 8 to restore integer result.

        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(f) << 8);

        if constexpr (is_fp32_dest_acc_en) {
            sfpi::vFloat c = (x * y) * (y * y);
            y = y * (c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0);

            sfpi::vFloat d = x * (y * y);
            c = d * y + sfpi::vConstNeg1;
            sfpi::vFloat negative_third = sfpi::addexp(negative_third_256, 8);
            sfpi::vFloat t = c * negative_third + sfpi::vConst1;
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

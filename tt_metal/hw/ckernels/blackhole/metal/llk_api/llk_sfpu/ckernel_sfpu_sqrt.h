// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, int RECIPROCAL_ITERATIONS = 2>
inline void calculate_sqrt() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];

        if constexpr (APPROXIMATION_MODE) {
            vUInt magic = vConstIntPrgm0;

            // sqrt initial approximation
            //  adjust bias
            vUInt val_s = magic + reinterpret<vUInt>(val);

            // approximation of square root
            val_s >>= 1;
            dst_reg[0] = reinterpret<vFloat>(val_s);
        } else {
            // Recip root method
            //// Init approx
            // u.i = SQRT_MAGIC_F - (u.i >> 1);
            v_if(val != 0.0f) {
                vUInt magic = vConstIntPrgm0;
                vFloat approx = reinterpret<vFloat>(magic - (reinterpret<vUInt>(val) >> 1));

                // Reciproot iterations
                for (int r = 0; r < RECIPROCAL_ITERATIONS; r++) {
                    // x*r*(1.5f - xhalf*r*r);
                    approx = ((approx * approx) * (val * -0.5f) + 1.5f) * approx;
                }

                dst_reg[0] = approx * val;
            }
            v_endif;
        }

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void sqrt_init() {
    if (APPROXIMATION_MODE) {
        vConstFloatPrgm0 = s2vFloat16b(127 << 7);
    } else {
        vConstFloatPrgm0 = s2vFloat16b(0x5f37);
    }
}
}  // namespace sfpu
}  // namespace ckernel

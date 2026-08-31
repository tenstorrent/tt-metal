// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

#define POLYVAL10(coef10, coef9, coef8, coef7, coef6, coef5, coef4, coef3, coef2, coef1, coef0, t4)               \
    ((coef0 +                                                                                                     \
      (coef1 +                                                                                                    \
       (coef2 +                                                                                                   \
        (coef3 +                                                                                                  \
         (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4) * t4) * t4) * \
            t4) *                                                                                                 \
           t4) *                                                                                                  \
          t4) *                                                                                                   \
     t4)
// Park the three costliest series coefficients in the SFPU's programmable constant
// registers: SFPMAD names a CREG directly in an operand field, so each costs zero
// instructions per element instead of the two SFPLOADI a full-fp32 literal needs, and the
// value keeps its exact fp32 bit pattern (bit-exact, not an accuracy trade). The other
// coefficients stay literals -- 0.25f and 0.015625f are already bf16-exact and free, and
// on Wormhole a dependent Horner chain hides some load latency, so draining every load
// out of the chain trades loads for SFPNOPs rather than removing them outright.
inline void i0_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = 0.0004340277778f;
    sfpi::vConstFloatPrgm1 = 0.000006781684028f;
    sfpi::vConstFloatPrgm2 = 6.78E-08f;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
#pragma GCC unroll 0

    for (int d = 0; d < ITERATIONS; d++) {
        vFloat result = 0.0f;
        vFloat input = dst_reg[0];
        vFloat x = input * input;

        result = 1.0f + POLYVAL10(
                            1.50E-22f,
                            7.24E-20f,
                            2.90E-17f,
                            9.39E-15f,
                            2.40E-12f,
                            4.71E-10f,
                            sfpi::vConstFloatPrgm2,  // 6.78E-08f,          parked by i0_init
                            sfpi::vConstFloatPrgm1,  // 0.000006781684028f, parked by i0_init
                            sfpi::vConstFloatPrgm0,  // 0.0004340277778f,   parked by i0_init
                            0.015625f,
                            0.25f,
                            x);

        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

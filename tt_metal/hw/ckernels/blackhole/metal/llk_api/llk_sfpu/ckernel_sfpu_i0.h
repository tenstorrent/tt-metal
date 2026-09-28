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
#ifndef INP_FLOAT32
// BF16-input arm: the same degree-11 Taylor series in t = x² (coefficient k = 1/(4^k (k!)²)), evaluated by explicit
// Horner in the order POLYVAL10 uses. I0_BF16_C3/C4 live in vConstFloatPrgm1/2 (programmed by i0_init) instead of
// costing two SFPLOADI per row each.
constexpr float I0_BF16_C3 = 0.0004340277778f;    // vConstFloatPrgm1
constexpr float I0_BF16_C4 = 0.000006781684028f;  // vConstFloatPrgm2
constexpr float I0_BF16_C5 = 6.78E-08f;
constexpr float I0_BF16_C6 = 4.71E-10f;
constexpr float I0_BF16_C7 = 2.40E-12f;
constexpr float I0_BF16_C8 = 9.39E-15f;
constexpr float I0_BF16_C9 = 2.90E-17f;
constexpr float I0_BF16_C10 = 7.24E-20f;
constexpr float I0_BF16_C11 = 1.50E-22f;
#endif

inline void i0_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef INP_FLOAT32
    vConstFloatPrgm1 = I0_BF16_C3;
    vConstFloatPrgm2 = I0_BF16_C4;
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
#pragma GCC unroll 0

    for (int d = 0; d < ITERATIONS; d++) {
        vFloat input = dst_reg[0];
        vFloat x = input * input;

#ifdef INP_FLOAT32
        vFloat result = 1.0f + POLYVAL10(
                                   1.50E-22f,
                                   7.24E-20f,
                                   2.90E-17f,
                                   9.39E-15f,
                                   2.40E-12f,
                                   4.71E-10f,
                                   6.78E-08f,
                                   0.000006781684028f,
                                   0.0004340277778f,
                                   0.015625f,
                                   0.25f,
                                   x);
#else
        vFloat result = I0_BF16_C11 * x + I0_BF16_C10;
        result = result * x + I0_BF16_C9;
        result = result * x + I0_BF16_C8;
        result = result * x + I0_BF16_C7;
        result = result * x + I0_BF16_C6;
        result = result * x + I0_BF16_C5;
        result = result * x + vConstFloatPrgm2;  // C4
        result = result * x + vConstFloatPrgm1;  // C3
        result = result * x + 0.015625f;
        result = result * x + 0.25f;
        result = result * x + 1.0f;
#endif

        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

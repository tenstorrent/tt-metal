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
// BF16-input arm: degree-11 polynomial in t = x², evaluated by explicit Horner in the order POLYVAL10 uses. It
// starts from the Taylor series (coefficient k = 1/(4^k (k!)²)); C3 and C4 are full fp32 constants held in
// vConstFloatPrgm1/2 (programmed by i0_init), C5 is an fp32 immediate, and C6..C11 are on the one-SFPLOADI bf16 grid,
// grid-searched against every bf16 input with both BF16 (truncating store) and FP32 dest so that max and mean ULP vs
// float64 do not regress anywhere. The INP_FLOAT32 arm keeps the exact series coefficients.
constexpr float I0_BF16_C3 = 0x1.c71c72p-12f;  // vConstFloatPrgm1
constexpr float I0_BF16_C4 = 0x1.c7238ep-18f;  // vConstFloatPrgm2
constexpr float I0_BF16_C5 = 0x1.234e32p-24f;
constexpr float I0_BF16_C6 = 0x1.02p-31f;
constexpr float I0_BF16_C7 = 0x1.56p-39f;
constexpr float I0_BF16_C8 = 0x1.4cp-47f;
constexpr float I0_BF16_C9 = 0x1.12p-55f;
constexpr float I0_BF16_C10 = 0x1.5cp-64f;
constexpr float I0_BF16_C11 = 0x1.6cp-73f;
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

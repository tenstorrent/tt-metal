// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

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
template <bool APPROXIMATION_MODE>
inline void calculate_i0() {
#pragma GCC unroll 0

    for (int d = 0; d < 8; d++) {
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
                            6.78E-08f,
                            0.000006781684028f,
                            0.0004340277778f,
                            0.015625f,
                            0.25f,
                            x);

        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

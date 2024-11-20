// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;

namespace ckernel {

namespace sfpu {

#define POLYVAL10_I1(coef10, coef9, coef8, coef7, coef6, coef5, coef4, coef3, coef2, coef1, coef0, t2)            \
    ((coef0 +                                                                                                     \
      (coef1 +                                                                                                    \
       (coef2 +                                                                                                   \
        (coef3 +                                                                                                  \
         (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t2) * t2) * t2) * t2) * t2) * t2) * t2) * \
            t2) *                                                                                                 \
           t2) *                                                                                                  \
          t2) *                                                                                                   \
     t2)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i1() {
#pragma GCC unroll 0

    for (int d = 0; d < ITERATIONS; d++) {
        vFloat result = 0.0f;
        vFloat input = dst_reg[0];
        vFloat x = input * input;

        vFloat derivative = input * POLYVAL10_I1(
                                        1.24695e-23f,
                                        6.58387e-21f,
                                        2.8969e-18f,
                                        1.04289e-15f,
                                        3.00351e-13f,
                                        6.72786e-11f,
                                        1.13028e-08f,
                                        1.35634e-06f,
                                        0.000108507f,
                                        0.00520833f,
                                        0.125f,
                                        x);
        result = input * 0.5f + derivative;
        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

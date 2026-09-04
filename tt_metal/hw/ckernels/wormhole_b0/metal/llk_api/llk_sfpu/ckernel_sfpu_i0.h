// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "llk_math_eltwise_sfpu_op.h"

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

// ---------------------------------------------------------------------------------------------------
// I0<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode) -> calculate_i0
//   init()                            -> shared SFPU init only
// Backs i0_tile / i0_tile_init.
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct I0 : SfpuUnaryOp<I0<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_i0<APPROXIMATION_MODE, ITERATIONS>(); }
};
}  // namespace sfpu
}  // namespace ckernel

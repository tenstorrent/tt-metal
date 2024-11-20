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

#define POLYVAL10_DERIVATIVE(coef10, coef9, coef8, coef7, coef6, coef5, coef4, coef3, coef2, coef1, coef0, t4)  \
    ((coef1 + \
    (2.0f * coef2 + \
    (3.0f * coef3 + \
    (4.0f * coef4 + \
    (5.0f * coef5 + \
    (6.0f * coef6 + \
    (7.0f * coef7 + \
    (8.0f * coef8 + \
    (9.0f * coef9 + \
    (10.0f * coef10 * t4) * t4) * t4) * t4) * t4) * t4) * t4) * t4) * t4) * t4) )

template <bool APPROXIMATION_MODE>
inline void calculate_i1()
{
    #pragma GCC unroll 0

    for (int d = 0; d < 8; d++)
    {
        vFloat result = 0.0f;
        vFloat input = dst_reg[0];
        vFloat x = input * input;

        vFloat derivative =  POLYVAL10_DERIVATIVE(1.50E-22f, 7.24E-20f, 2.90E-17f, 9.39E-15f, 2.40E-12f, 4.71E-10f,
                                                 6.78E-08f, 0.000006781684028f, 0.0004340277778f, 0.015625f,
                                                 0.25f, x);
        result = 2.0f * input * derivative;

        dst_reg[0] = result;
        dst_reg++;
    }

}

}  // namespace sfpu
}  // namespace ckernel

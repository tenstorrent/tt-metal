// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <int max_iter = 3>
sfpi_inline vFloat _sfpu_reciprocal_(const vFloat in)
{
    // Force sign to 1 (make number negative)
    vFloat val = setsgn(in, 1);

    val = setexp(val, 126); // Set exponent to 126 to make the number in 0.5-1
    // Use 1.44 as first guess at x, ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
    vFloat vConstLn2Recip = vConstFloatPrgm0;
    vFloat two = vConstFloatPrgm1;
    vFloat result = vConstLn2Recip * (val * vConstLn2Recip + two);

    for (int s_iter = 0; s_iter < (max_iter-1); s_iter++) {
        result = result * (val * result + two);
    }

    vInt orig_exp = exexp(in);
    vInt new_exp = exexp(result);

    // "Subtract" exponents, and re-bias.
    // Execute: -1 - exp, then exp += 127
    new_exp -= orig_exp;
    new_exp += 126;

    v_if (new_exp < 0) {
        // If rebiased exponent is negative, we need to saturate at 0.
        // This means the initial number was too big so reciprocal result should be 0
        result = 0.0F;
        new_exp = 0;
    }
    v_endif;

    // Set newly denormalized exponent to result exponent field
    return setexp(result, new_exp);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_reciprocal_(const int iterations)
{
    #pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        vFloat in = dst_reg[0];
        vFloat out = _sfpu_reciprocal_<APPROXIMATION_MODE ? 2 : 3>(in);

        v_if (in < 0.0F) {
            // Invert sign on calculated value if CC=1 (number is negative)
            out = -out;
        }
        v_endif;

        dst_reg[0] = out;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_reciprocal_()
{
    vConstFloatPrgm0 = 1.442695f; // ln2_recip
    vConstFloatPrgm1 = 2.0f;
}

} // namespace sfpu
} // namespace ckernel

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <int max_iter = 3>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in)
{
    // Force sign to 1 (make number negative)
    sfpi::vFloat val = sfpi::setsgn(in, 1);

    val                 = setexp(val, 126); // Set exponent to 126 to make the number in 0.5-1
    sfpi::vFloat a      = sfpi::vConstFloatPrgm0;
    sfpi::vFloat b      = sfpi::vConstFloatPrgm1;
    sfpi::vFloat result = a + b * val;

    for (int s_iter = 0; s_iter < max_iter; s_iter++)
    {
        result += result * (val * result + sfpi::vConst1);
    }

    sfpi::vInt orig_exp = exexp(in);
    sfpi::vInt new_exp  = exexp(result);
    new_exp -= orig_exp;
    new_exp += 126;

    v_if (new_exp >= 0)
    {
        // Set newly denormalized exponent to result exponent field
        result = setexp(result, new_exp);
    }
    v_else
    {
        // If rebiased exponent is negative, we need to saturate at 0.
        // This means the initial number was too big so reciprocal result should be 0
        result = 0.0f;
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en = true>
inline void _calculate_reciprocal_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat in  = sfpi::dst_reg[0];
        sfpi::vFloat out = _sfpu_reciprocal_<APPROXIMATION_MODE ? 2 : 3>(in);
        out              = setsgn(out, sfpi::reinterpret<sfpi::vInt>(in));

        if constexpr (is_fp32_dest_acc_en || APPROXIMATION_MODE)
        {
            sfpi::dst_reg[0] = out;
        }
        else
        {
            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(out, 0));
        }

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_reciprocal_()
{
    // The following constants are used to calculate an initial estimate for 1/D using a linear approximation.
    // The linear approximation with minimum worst-case absolute error on the interval [0.5, 1] is:
    //   X_0 = 48/17 - 32/17 D
    // See https://en.wikipedia.org/wiki/Division_algorithm#Initial_estimate for the full derivation.

    sfpi::vConstFloatPrgm0 = 48.0f / 17.0f;
    sfpi::vConstFloatPrgm1 = 32.0f / 17.0f;
}

} // namespace sfpu
} // namespace ckernel

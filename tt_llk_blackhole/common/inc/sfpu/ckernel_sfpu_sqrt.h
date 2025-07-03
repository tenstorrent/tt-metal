// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int RECIPROCAL_ITERATIONS>
sfpi_inline sfpi::vFloat _calculate_sqrt_body_(sfpi::vFloat val)
{
    sfpi::vFloat result;
    if constexpr (APPROXIMATION_MODE)
    {
        sfpi::vUInt magic = sfpi::vConstIntPrgm0;

        // sqrt initial approximation
        //  adjust bias
        sfpi::vUInt val_s = magic + sfpi::reinterpret<sfpi::vUInt>(val);

        // approximation of square root
        val_s >>= 1;
        result = sfpi::reinterpret<sfpi::vFloat>(val_s);
    }
    else
    {
        // Recip root method
        //// Init approx
        // u.i = SQRT_MAGIC_F - (u.i >> 1);
        v_if (val != 0.0f)
        {
            sfpi::vUInt magic   = sfpi::vConstIntPrgm0;
            sfpi::vFloat approx = sfpi::reinterpret<sfpi::vFloat>(magic - (sfpi::reinterpret<sfpi::vUInt>(val) >> 1));

            // Reciproot iterations
            for (int r = 0; r < RECIPROCAL_ITERATIONS; r++)
            {
                // x*r*(1.5f - xhalf*r*r);
                approx = ((approx * approx) * (val * -0.5f) + 1.5f) * approx;
            }

            result = approx * val;
        }
        v_else
        {
            result = val;
        }
        v_endif;
    }
    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, int RECIPROCAL_ITERATIONS>
inline void _calculate_sqrt_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _calculate_sqrt_body_<APPROXIMATION_MODE, RECIPROCAL_ITERATIONS>(val);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_sqrt_()
{
    if (APPROXIMATION_MODE)
    {
        sfpi::vConstFloatPrgm0 = sfpi::s2vFloat16b(127 << 7);
    }
    else
    {
        sfpi::vConstFloatPrgm0 = sfpi::s2vFloat16b(0x5f37);
    }
}

} // namespace sfpu
} // namespace ckernel

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_rsqrt_compat.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// Computes the reciprocal of a floating point value x.
// max_iter specifies the number of Newton-Raphson iterations.
// max_iter = 2: sufficient for float32 precision (≤1 ulps).
// max_iter = 1: sufficient for bfloat16/float16 precision (≤0.5 ulps).
// max_iter = 0: this has the same effect as max_iter=1 at the moment;
//               it may be replaced with a cheaper approximation in future.
template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in)
{
    // Combines the sign and exponent of -1.0 with the mantissa of `in`.
    // Scale the input value to the range [1.0, 2.0), and make it negative.
    // If in ≠ ±0 and in ≠ ±inf, then x = in * 2**(127-in.Exp).
    // If in = ±0 or in = ±inf, then x = ±1.
    // Then negative_x = -x.
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in));

    // Quadratic initial estimate: y = k2 - k1*x + k0*x**2.
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;

    // Scale factor: we want 1/in = 1/x * scale.
    // For x ≠ ±0 and x ≠ ±inf, in = x * 2**-(127-in.Exp), so 1/in = 1/x * 2**(127-in.Exp).
    // Add float32 bias: scale.Exp = 127+127-in.Exp = 254-in.Exp.
    // For efficiency and handling of x = ±0 and x = ±inf, we set scale.Exp = 255-in.Exp = ~in.Exp.
    // This is efficiently computed with a single SFPNOT, followed by SFPSETMAN to clear the mantissa at the next opportunity.
    // The sign doesn't matter as we set the output sign to match the input at the end.
    // Not only is 255-in.Exp more efficient via SFPNOT, but it also ensures
    // that in.Exp == 0 results in ±inf, and in.Exp == 255 results in ±0.
    // See the scale factor adjustment via scale*0.5 below for further details.
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in);

    // Continue with quadratic estimate.
    y = sfpi::vConstFloatPrgm2 + y * negative_x;

    // Scale factor: set mantissa to zero.
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0);

    // First iteration of Newton-Raphson: t = 1.0 - x*y.
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;

    // Scale factor adjustment: scale = scale*0.5.
    // If scale = ±inf, then scale*0.5 = ±inf and scale.Exp=255.
    // If scale = ±0, then scale*0.5 = 0 and scale.Exp=0.
    // Otherwise, scale.Exp = scale.Exp-1 = 255-in.Exp-1 = 254-in.Exp.
    scale *= 0.5f;

    // Continue Newton-Raphson: y = y + y*t.
    y = y + y * t;

    if constexpr (max_iter > 1)
    {
        // Second iteration of Newton-Raphson: t = 1.0 - x*y; y = y + y*t.
        t = sfpi::vConst1 + negative_x * y;
        y = y + y * t;
    }

    // Apply scaling factor, and set sign to match input.
    y = y * scale;
    y = sfpi::setsgn(y, in);

    return y;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void _calculate_reciprocal_internal_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];

        if constexpr (APPROXIMATION_MODE)
        {
            sfpi::dst_reg[0] = _sfpu_reciprocal_<0>(in);
        }
        else
        {
            if constexpr (is_fp32_dest_acc_en)
            {
                sfpi::dst_reg[0] = _sfpu_reciprocal_<2>(in);
            }
            else
            {
                sfpi::vFloat out = _sfpu_reciprocal_<1>(in);
                sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(out, 0));
            }
        }

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en, bool legacy_compat = false>
inline void _calculate_reciprocal_(const int iterations)
{
    if constexpr (legacy_compat)
    {
        _calculate_reciprocal_compat_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    }
    else
    {
        _calculate_reciprocal_internal_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    }
}

template <bool APPROXIMATION_MODE, bool legacy_compat = false>
inline void _init_reciprocal_()
{
    if constexpr (!legacy_compat)
    {
        // The polynomial y = k2 - k1*x + k0*x**2 minimises the maximum
        // absolute error for 1/x over the interval [1,2), found via Sollya.
        sfpi::vConstFloatPrgm0 = 0.343145549297332763671875f;
        sfpi::vConstFloatPrgm1 = 1.51471805572509765625f;
        sfpi::vConstFloatPrgm2 = 2.1642131805419921875f;
    }
}

} // namespace sfpu
} // namespace ckernel

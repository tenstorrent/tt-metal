// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_rsqrt_compat.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel
{
namespace sfpu
{

// See: Kokosiński, Z., Gepner, P., Moroz, L. et al.
// Fast and accurate approximation algorithms for computing floating point square root. Numerical Algorithms (2024).
// https://doi.org/10.1007/s11075-024-01932-7

// Computes the square root or reciprocal square root of a positive floating point value x.
template <bool APPROXIMATE = false, bool RECIPROCAL = false>
sfpi_inline sfpi::vFloat _calculate_sqrt_body_(const sfpi::vFloat x)
{
    sfpi::vInt i   = sfpi::reinterpret<sfpi::vInt>(sfpi::reinterpret<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(sfpi::vConstIntPrgm0 - i);

    if constexpr (APPROXIMATE)
    {
        // Algorithm SQRT_10-bits, with modifications for reciprocal.
        sfpi::vFloat c           = x * y;
        sfpi::vFloat negative_y  = -y;
        sfpi::vFloat infinity    = sfpi::s2vFloat16b(std::numeric_limits<float>::infinity());
        sfpi::vInt infinity_bits = sfpi::reinterpret<sfpi::vInt>(infinity);
        sfpi::vFloat t           = sfpi::vConstFloatPrgm1 + negative_y * c;
        if constexpr (RECIPROCAL)
        {
            sfpi::vInt x_bits                = sfpi::reinterpret<sfpi::vInt>(x);
            sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
            // If x != inf and x != 0.
            v_if (infinity_minus_x_bits != 0 && x_bits != 0)
            {
                y = y * t;
            }
            // Otherwise, if x = 0, then y = inf; if x = inf, then y = 0.
            v_else
            {
                y = sfpi::reinterpret<sfpi::vFloat>(infinity_minus_x_bits);
            }
            v_endif;
        }
        else
        {
            y = c;
            // If x != inf.  Otherwise, y = inf, since c = inf.
            v_if (sfpi::reinterpret<sfpi::vInt>(x) != infinity_bits)
            {
                y = y * t;
            }
            v_endif;
        }
    }
    else
    {
        // Algorithm SQRT_23-bits, with modifications for reciprocal.
        sfpi::vFloat xy            = x * y;
        sfpi::vFloat negative_y    = -y;
        sfpi::vFloat c             = negative_y * xy;
        sfpi::vFloat infinity      = sfpi::s2vFloat16b(std::numeric_limits<float>::infinity());
        sfpi::vInt infinity_bits   = sfpi::reinterpret<sfpi::vInt>(infinity);
        y                          = y * (sfpi::vConstFloatPrgm1 + c * (sfpi::vConstFloatPrgm2 + c));
        xy                         = x * y;
        negative_y                 = -y;
        sfpi::vFloat one_minus_xyy = sfpi::vConst1 + (negative_y * xy);

        if constexpr (RECIPROCAL)
        {
            sfpi::vFloat half_y              = sfpi::addexp(y, -1);
            sfpi::vInt x_bits                = sfpi::reinterpret<sfpi::vInt>(x);
            sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
            // If x != inf and x != 0.
            v_if (infinity_minus_x_bits != 0 && x_bits != 0)
            {
                y = one_minus_xyy * half_y + y;
            }
            // Otherwise, if x = 0, then y = inf; if x = inf, then y = 0.
            v_else
            {
                y = sfpi::reinterpret<sfpi::vFloat>(infinity_minus_x_bits);
            }
            v_endif;
        }
        else
        {
            sfpi::vFloat half_xy = 0.5f * xy;
            // If x == inf, we need to skip to avoid y = inf - inf = nan; y will already be inf.
            v_if (sfpi::reinterpret<sfpi::vInt>(x) < infinity_bits)
            {
                y = one_minus_xyy * half_xy + xy;
            }
            v_endif;
        }
    }

    return y;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en, bool RECIPROCAL>
inline void _calculate_sqrt_internal_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat tmp = _calculate_sqrt_body_<APPROXIMATION_MODE, RECIPROCAL>(sfpi::dst_reg[0]);
        if constexpr (fp32_dest_acc_en)
        {
            sfpi::dst_reg[0] = tmp;
        }
        else
        {
            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(tmp, 0));
        }
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en, bool legacy_compat = false>
inline void _calculate_sqrt_(int iterations)
{
    if constexpr (legacy_compat)
    {
        return _calculate_sqrt_compat_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en>(iterations);
    }
    else
    {
        return _calculate_sqrt_internal_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, false>(iterations);
    }
}

template <bool APPROXIMATION_MODE, bool legacy_compat = false>
inline void _init_sqrt_()
{
    if constexpr (!legacy_compat)
    {
        if constexpr (APPROXIMATION_MODE)
        {
            sfpi::vConstIntPrgm0   = 0x5f0b3892;
            sfpi::vConstFloatPrgm1 = 1.89099014875f;
        }
        else
        {
            sfpi::vConstIntPrgm0   = 0x5f1110a0;
            sfpi::vConstFloatPrgm1 = 2.2825186f;
            sfpi::vConstFloatPrgm2 = 2.2533049f;
        }
    }
}

} // namespace sfpu
} // namespace ckernel

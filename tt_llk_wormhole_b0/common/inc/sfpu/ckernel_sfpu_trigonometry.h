// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_sqrt.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _sfpu_sine_maclaurin_series_(sfpi::vFloat val)
{
    // Good for [-pi:pi]
    // Mclauren series = x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    sfpi::vFloat tmp = val;
    // x
    sfpi::vFloat output = tmp;
    // x^3/3!
    tmp = tmp * val * val;
    output += -0.166666666 * tmp;
    // x^5/5!
    tmp = tmp * val * val;
    output += 0.0083333333 * tmp;
    // x^7/7!
    tmp = tmp * val * val;
    output += -0.0001984126 * tmp;
    if constexpr (not APPROXIMATION_MODE)
    {
        // x^9/9!
        tmp = tmp * val * val;
        output += 0.0000027557 * tmp;
        // x^11/11!
        tmp = tmp * val * val;
        output += -0.00000002505 * tmp;
    }

    // Write out output
    return output;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _sfpu_cosine_maclaurin_series_(sfpi::vFloat val)
{
    // Good for [-pi:pi]
    // Mclauren series = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! - x^10/10! + x^12/12!
    // 1
    sfpi::vFloat output = 1.0f;
    // x^2/2!
    sfpi::vFloat tmp = val * val;
    output += -0.5 * tmp;
    // x^4/4!
    tmp = tmp * val * val;
    output += 0.0416666666 * tmp;
    // x^6/6!
    tmp = tmp * val * val;
    output += -0.0013888888 * tmp;
    if constexpr (not APPROXIMATION_MODE)
    {
        // x^8/8!
        tmp = tmp * val * val;
        output += 0.0000248015 * tmp;
        // x^10/10!
        tmp = tmp * val * val;
        output += -0.0000002755 * tmp;
    }

    // Write out output
    return output;
}

// Legacy implementation.
// Candidate for removal in future versions. See https://github.com/tenstorrent/tt-llk/issues/225 for more details.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sine_(const int iterations)
{
    // SFPU microcode
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat v             = sfpi::dst_reg[0];
        v                          = 0.318309886183791f * v; // *1/pi to get number of pi rads.
        sfpi::vInt whole_v         = float_to_int16(v, 0);
        sfpi::vFloat whole_v_float = int32_to_float(whole_v, 0);
        v                          = v - whole_v_float;
        v *= 3.141592653589793f; // fractional * pi to get it in [-pi:pi]
        v       = _sfpu_sine_maclaurin_series_<APPROXIMATION_MODE>(v);
        whole_v = whole_v & 0x1;
        v_if (whole_v != 0)
        {
            // odd so flip the sign
            v *= -1;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

// Legacy implementation.
// Candidate for removal in future versions. See https://github.com/tenstorrent/tt-llk/issues/225 for more details.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_cosine_(const int iterations)
{
    // SFPU microcode
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat v             = sfpi::dst_reg[0];
        v                          = 0.318309886183791f * v; // *1/pi to get number of pi rads.
        sfpi::vInt whole_v         = float_to_int16(v, 0);
        sfpi::vFloat whole_v_float = int32_to_float(whole_v, 0);
        v                          = v - whole_v_float;
        v *= 3.141592653589793f; // fractional * pi to get it in [-pi:pi]
        v       = _sfpu_cosine_maclaurin_series_<APPROXIMATION_MODE>(v);
        whole_v = whole_v & 0x1;
        v_if (whole_v != 0)
        {
            // odd so flip the sign
            v *= -1;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

// https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#Definitions_in_terms_of_logarithms
// acosh(x) = log(x + sqrt(x^2 - 1))
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_acosh_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat inp = sfpi::dst_reg[0];
        v_if (inp < sfpi::vConst1)
        {
            sfpi::dst_reg[0] = std::numeric_limits<float>::quiet_NaN();
        }
        v_elseif (inp == sfpi::vConst1)
        {
            sfpi::dst_reg[0] = sfpi::vConst0;
        }
        v_else
        {
            sfpi::vFloat tmp = inp * inp;
            tmp              = tmp - sfpi::vConst1;
            tmp              = _calculate_sqrt_body_<APPROXIMATION_MODE, 2>(tmp);
            tmp              = tmp + inp;
            sfpi::dst_reg[0] = _calculate_log_body_no_init_(tmp);
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

// asinh(x) = log(x + sqrt(x^2 + 1))
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_asinh_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat inp = sfpi::dst_reg[0];
        sfpi::vFloat tmp = inp * inp + sfpi::vConst1;
        tmp              = _calculate_sqrt_body_<APPROXIMATION_MODE, 2>(tmp);
        tmp              = tmp + sfpi::abs(inp);
        sfpi::dst_reg[0] = _calculate_log_body_no_init_(tmp);
        v_if (inp < sfpi::vConst0)
        {
            sfpi::dst_reg[0] = -sfpi::dst_reg[0];
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

// atanh[x] = 0.5 * ln((1 + x) / (1 - x))
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void _calculate_atanh_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat inp     = sfpi::dst_reg[0];
        sfpi::vFloat abs_inp = sfpi::abs(inp);
        v_if (abs_inp > sfpi::vConst1)
        {
            sfpi::dst_reg[0] = std::numeric_limits<float>::quiet_NaN();
        }
        v_elseif (abs_inp == sfpi::vConst1)
        {
            sfpi::vFloat inf = std::numeric_limits<float>::infinity();
            sfpi::dst_reg[0] = sfpi::setsgn(inf, inp);
        }
        v_else
        {
            sfpi::vFloat num = sfpi::vConst1 + inp;
            sfpi::vFloat den = sfpi::vConst1 - inp;
            sfpi::vFloat tmp = _sfpu_reciprocal_<APPROXIMATION_MODE ? 2 : 3>(den);
            tmp              = sfpi::setsgn(tmp, den);
            if constexpr (is_fp32_dest_acc_en || APPROXIMATION_MODE)
            {
                den = tmp;
            }
            else
            {
                den = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(tmp, 0));
            }
            num              = num * den;
            den              = _calculate_log_body_no_init_(num);
            sfpi::dst_reg[0] = 0.5f * den;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void _init_inverse_hyperbolic_()
{
    _init_sqrt_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
void _init_atanh_()
{
    _init_reciprocal_<APPROXIMATION_MODE>();
}

} // namespace sfpu
} // namespace ckernel

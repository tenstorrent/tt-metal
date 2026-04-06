// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Reciprocal using Newton-Raphson iteration from SFPI instructions.
// Computes 1/in for positive in. Sign must be handled by caller.
template <int max_iter = 2>
sfpi_inline sfpi::vFloat _lgamma_reciprocal_(const sfpi::vFloat in) {
    // Force sign to positive
    sfpi::vFloat val = sfpi::setsgn(in, 0);

    // Save original exponent
    sfpi::vInt orig_exp = sfpi::exexp(val);

    // Normalize to [0.5, 1.0) range by setting exponent to 126
    val = sfpi::setexp(val, 126);

    // Initial guess: 1.44 * (val * 1.44 + 2.0)
    // This is the standard Newton-Raphson seed for reciprocal
    sfpi::vFloat vConstLn2Recip = 1.442695f;
    sfpi::vFloat two = 2.0f;
    sfpi::vFloat result = vConstLn2Recip * (val * vConstLn2Recip + two);

    // Newton-Raphson iterations: result = result * (val * result + 2)
    for (int s_iter = 0; s_iter < (max_iter - 1); s_iter++) {
        result = result * (val * result + two);
    }

    // Reconstruct exponent: new_exp = new_exp - orig_exp + 126
    sfpi::vInt new_exp = sfpi::exexp(result);
    new_exp -= orig_exp;
    new_exp += 126;

    v_if(new_exp < 0) {
        result = 0.0f;
        new_exp = 0;
    }
    v_endif;

    return sfpi::setexp(result, new_exp);
}

// Natural logarithm using exponent extraction + polynomial approximation.
// Computes ln(x) for x > 0. Undefined for x <= 0.
sfpi_inline sfpi::vFloat _lgamma_log_(const sfpi::vFloat x) {
    // Extract debiased exponent: for x = m * 2^e, gives e
    sfpi::vInt exp_i = sfpi::exexp(x);

    // Set exponent to 127 to get mantissa m in [1.0, 2.0)
    sfpi::vFloat m = sfpi::setexp(x, 127);

    // f = m - 1.0, so f in [0, 1)
    sfpi::vFloat f = m - sfpi::vConst1;

    // Polynomial approximation of ln(1+f) using Horner's method:
    // ln(1+f) ~ f - f^2/2 + f^3/3 - f^4/4 + f^5/5
    // = f * (1 + f * (-0.5 + f * (0.3333 + f * (-0.25 + f * 0.2))))
    sfpi::vFloat poly = f * 0.2f;
    poly = poly + -0.25f;
    poly = poly * f;
    poly = poly + 0.333333f;
    poly = poly * f;
    poly = poly + -0.5f;
    poly = poly * f;
    poly = poly + sfpi::vConst1;
    poly = poly * f;

    // Convert exponent to float
    sfpi::vFloat exp_f = sfpi::int32_to_float(exp_i);

    // result = exponent * ln(2) + ln(mantissa)
    return exp_f * 0.6931472f + poly;
}

// lgamma(x) = ln(|Gamma(x)|)
// Uses the Lanczos approximation with g=5, matching the existing composite implementation.
//
// For x > 0:
//   z = x - 1
//   ser = 1.0 + 76.18009/(z+1) - 86.50532/(z+2) + 24.01410/(z+3)
//             - 1.231740/(z+4) + 0.001209/(z+5) - 0.000005395/(z+6)
//   tmp = z + 5.5
//   lgamma = (z + 0.5) * ln(tmp) - tmp + ln(sqrt(2*pi)) + ln(ser)
//
// Special cases: lgamma(1) = 0, lgamma(2) = 0
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lgamma() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // z = x - 1
        sfpi::vFloat z = x - sfpi::vConst1;

        // Compute Lanczos series sum:
        // ser = 1.0 + c1/(z+1) + c2/(z+2) + c3/(z+3) + c4/(z+4) + c5/(z+5)
        // Note: z+1 = x, z+2 = x+1, etc.
        sfpi::vFloat ser = sfpi::vConst1;

        // Term 1: 76.18009172947146 / (z + 1)
        sfpi::vFloat denom = z + sfpi::vConst1;
        sfpi::vFloat recip = _lgamma_reciprocal_<2>(denom);
        ser = ser + recip * 76.18009f;

        // Term 2: -86.50532032941677 / (z + 2)
        denom = z + 2.0f;
        recip = _lgamma_reciprocal_<2>(denom);
        ser = ser + recip * -86.50532f;

        // Term 3: 24.01409824083091 / (z + 3)
        denom = z + 3.0f;
        recip = _lgamma_reciprocal_<2>(denom);
        ser = ser + recip * 24.01410f;

        // Term 4: -1.231739572450155 / (z + 4)
        denom = z + 4.0f;
        recip = _lgamma_reciprocal_<2>(denom);
        ser = ser + recip * -1.231740f;

        // Term 5: 0.1208650973866179e-2 / (z + 5)
        denom = z + 5.0f;
        recip = _lgamma_reciprocal_<2>(denom);
        ser = ser + recip * 0.001209f;

        // Term 6: -0.5395239384953e-5 / (z + 6)
        // This term is negligible for bfloat16 precision, skip it.

        // tmp = z + 5.5
        sfpi::vFloat tmp = z + 5.5f;

        // log_tmp = ln(tmp)
        sfpi::vFloat log_tmp = _lgamma_log_(tmp);

        // log_ser = ln(ser)
        sfpi::vFloat log_ser = _lgamma_log_(ser);

        // result = (z + 0.5) * log(tmp) - tmp + ln(sqrt(2*pi)) + log(ser)
        // ln(sqrt(2*pi)) = 0.9189385332046727
        sfpi::vFloat result = (z + 0.5f) * log_tmp;
        result = result - tmp;
        result = result + 0.918939f;
        result = result + log_ser;

        // Special cases: lgamma(1) = 0, lgamma(2) = 0
        v_if(x == sfpi::vConst1) { result = sfpi::vConst0; }
        v_endif;

        v_if(x == 2.0f) { result = sfpi::vConst0; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

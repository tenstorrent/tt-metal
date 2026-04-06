// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// atanh(x) = 0.5 * ln((1+x)/(1-x))
//
// Two-path implementation to maintain precision across the full input range:
//
//   Path A (|x| < 0.5): Taylor series
//     atanh(x) ≈ x + x³/3 + x⁵/5
//     Used to avoid catastrophic cancellation when (1+x)/(1-x) ≈ 1
//
//   Path B (|x| >= 0.5): Full logarithm computation
//     1. Compute numerator = 1 + x, denominator = 1 - x
//     2. Compute reciprocal of denominator using Newton-Raphson
//     3. Compute ratio = numerator * recip(denominator)
//     4. Compute ln(ratio) using exponent extraction + polynomial approximation
//     5. Multiply by 0.5

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
    // Polynomial coefficients for ln(m) over m in [1, 2)
    // Coefficients from ckernel_sfpu_unary_power.h (proven in rpow kernel)
    constexpr float a0 = -0x1.952992p+0f;  // ≈ -1.5828
    constexpr float a1 = 0x2.4f5388p+0f;   // ≈  2.3097
    constexpr float a2 = -0xd.e712ap-4f;   // ≈ -0.8691
    constexpr float a3 = 0x2.44734p-4f;    // ≈  0.1419
    constexpr float ln2 = 0.6931471805599453f;
    constexpr float one_third = 0.333333343f;
    constexpr float one_fifth = 0.200000003f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat abs_x = sfpi::abs(x);

        // Default path: Taylor series for small |x|
        // atanh(x) ≈ x + x³/3 + x⁵/5
        sfpi::vFloat x2 = x * x;
        sfpi::vFloat x3 = x2 * x;
        sfpi::vFloat x5 = x3 * x2;
        sfpi::vFloat result = x + x3 * one_third + x5 * one_fifth;

        // Override for |x| >= 0.5: use full log computation
        v_if(abs_x >= 0.5f) {
            // Step 1: numerator and denominator
            sfpi::vFloat numer = x + sfpi::vConst1;  // 1 + x
            sfpi::vFloat denom = sfpi::vConst1 - x;  // 1 - x

            // Step 2: reciprocal of denom via Newton-Raphson
            sfpi::vFloat abs_denom = sfpi::abs(denom);
            sfpi::vInt denom_exp = sfpi::exexp(abs_denom);
            sfpi::vFloat denom_mant = sfpi::setexp(abs_denom, 127);

            // Linear initial estimate: 1/m ≈ 2.8235294 - 1.8823529*m for m in [1,2)
            sfpi::vFloat r = denom_mant * -1.8823529f + 2.8235294f;

            // Three Newton-Raphson iterations
            sfpi::vFloat nr_term = denom_mant * r;
            r = r * (2.0f - nr_term);
            nr_term = denom_mant * r;
            r = r * (2.0f - nr_term);
            nr_term = denom_mant * r;
            r = r * (2.0f - nr_term);

            // Adjust exponent and sign
            r = sfpi::addexp(r, -denom_exp);
            r = sfpi::setsgn(r, denom);

            // Step 3: ratio = numer / denom
            sfpi::vFloat ratio = numer * r;

            // Step 4: ln(ratio)
            sfpi::vFloat abs_ratio = sfpi::abs(ratio);
            sfpi::vInt ratio_exp = sfpi::exexp(abs_ratio);
            sfpi::vFloat ratio_mant = sfpi::setexp(abs_ratio, 127);

            // Polynomial: ln(m) ≈ a0 + m*(a1 + m*(a2 + m*a3))
            sfpi::vFloat ln_mant = ratio_mant * a3 + a2;
            ln_mant = ratio_mant * ln_mant + a1;
            ln_mant = ratio_mant * ln_mant + a0;

            sfpi::vFloat ratio_exp_f = sfpi::int32_to_float(ratio_exp, 0);
            sfpi::vFloat ln_ratio = ratio_exp_f * ln2 + ln_mant;

            // Step 5: result = 0.5 * ln(ratio)
            result = ln_ratio * 0.5f;
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

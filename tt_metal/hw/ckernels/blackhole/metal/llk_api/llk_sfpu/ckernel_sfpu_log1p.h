// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_log1p() {
    sfpi::vFloat a = sfpi::vConstFloatPrgm1;
    sfpi::vFloat b = sfpi::vConstFloatPrgm2;
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        in = in + 1.0f;
        sfpi::vFloat x = sfpi::setexp(in, 127);  // Normalize to [1, 2] range

        // Cheby Approximation using Horner Form Multiplication: 3rd Order
        sfpi::vFloat series_result = x * (x * (x * a + b) + 2.0871) + -1.4753f;

        sfpi::vInt exp = sfpi::exexp(in);
        v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
        v_endif;

        sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);
        sfpi::vFloat result = expf * vConstLn2 + series_result;

        v_if(in == 0.0F) { result = -std::numeric_limits<float>::infinity(); }
        v_endif;

        if constexpr (!FAST_APPROX) {
            v_if(in < 0.0F) {
                result = std::numeric_limits<float>::quiet_NaN();  // returns nan for fp32 and inf for bf16
            }
            v_endif;
        }

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;
        ++sfpi::dst_reg;
    }
}

/*
 * This function implements ln(1+x) with accurate computation for small x.
 * Algorithm based on log1p_fp32 from operations.py:
 * 1. Handle special cases (x < -1, x == -1, infinity, NaN)
 * 2. For |x| < 0.3: Use 13-term Taylor series to avoid catastrophic cancellation
 * 3. For |x| >= 0.3: Use standard ln(1+x) computation
 *
 * The threshold of 0.3 and 13 terms were chosen through systematic optimization
 * to minimize maximum ULP error across the entire domain.
 *
 * @param val The input value (vFloat vector), can be any floating point number > -1
 * @return vFloat Result of ln(1+val)
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log1p_fp32(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    // Constants
    constexpr float THRESHOLD = 0.3f;
    constexpr float NEG_ONE = -1.0f;
    constexpr float LN2 = 0.6931471805599453f;
    constexpr float SQRT2 = 1.4142135623730951f;
    constexpr float HALF = 0.5f;
    constexpr float ONE = 1.0f;
    constexpr float TWO = 2.0f;

    // Check for special cases using IEEE 754 bit patterns
    sfpi::vInt exp_bits = sfpi::exexp(val);   // Get debiased exponent
    sfpi::vInt man_bits = sfpi::exman9(val);  // Get mantissa bits (without implicit bit)

    // NaN check: debiased exponent == 128 (original exp = 255) and mantissa != 0
    v_if(exp_bits == 128 && man_bits != 0) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(val < NEG_ONE) {
        // Domain error: ln(1+x) undefined for x < -1
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_elseif(val == NEG_ONE) {
        // ln(0) = -infinity
        result = -std::numeric_limits<float>::infinity();
    }
    v_elseif(exp_bits == 128 && man_bits == 0 && val > sfpi::vConst0) {
        // Positive infinity: ln(+inf) = +inf
        result = std::numeric_limits<float>::infinity();
    }
    v_else {
        // Normal computation
        sfpi::vFloat abs_val = sfpi::abs(val);

        v_if(abs_val < THRESHOLD) {
            // For |x| < 0.3, use 13-term Taylor series to avoid catastrophic cancellation
            // ln(1+x) = x - x²/2 + x³/3 - x⁴/4 + ... + x¹³/13
            // Using PolynomialEvaluator with coefficients in ascending order:
            // ln(1+x) = 0 + x*(1 + x*(-1/2 + x*(1/3 + x*(-1/4 + ...))))
            result = sfpi::PolynomialEvaluator::eval(
                val,
                sfpi::vConst0,  // c0 = 0
                sfpi::vConst1,  // c1 = 1
                -0.5f,          // c2 = -1/2
                1.0f / 3.0f,    // c3 = 1/3
                -0.25f,         // c4 = -1/4
                0.2f,           // c5 = 1/5
                -1.0f / 6.0f,   // c6 = -1/6
                1.0f / 7.0f,    // c7 = 1/7
                -0.125f,        // c8 = -1/8
                1.0f / 9.0f,    // c9 = 1/9
                -0.1f,          // c10 = -1/10
                1.0f / 11.0f,   // c11 = 1/11
                -1.0f / 12.0f,  // c12 = -1/12
                1.0f / 13.0f    // c13 = 1/13
            );
        }
        v_else {
            // For |x| >= 0.3, use standard ln(1+x) computation
            // Apply the same algorithm as log-claude.cpp but on (1+x)
            sfpi::vFloat one_plus_x = sfpi::vConst1 + val;

            // Extract exponent (debiased) from (1+x)
            sfpi::vInt exp = sfpi::exexp(one_plus_x);

            // Extract mantissa and construct m in [1, 2)
            // Use setexp to normalize to [1, 2) range by setting exponent to 127 (bias)
            sfpi::vFloat m = sfpi::setexp(one_plus_x, 127);

            // Range reduction: if m >= sqrt(2), divide by 2 and increment exponent
            // This ensures m is in [sqrt(2)/2, sqrt(2)] ≈ [0.707, 1.414]
            v_if(m >= SQRT2) {
                m = m * HALF;
                exp = exp + 1;
            }
            v_endif;

            // Transform to z = (m - 1) / (m + 1)
            // This maps m ∈ [0.707, 1.414] to z ∈ [-0.172, 0.172]
            // ln(m) = 2 × (z + z³/3 + z⁵/5 + z⁷/7 + ...)
            sfpi::vFloat m_minus_1 = m - ONE;
            sfpi::vFloat m_plus_1 = m + ONE;

            // Compute z = (m - 1) / (m + 1) using reciprocal
            sfpi::vFloat m_plus_1_recip = ckernel::sfpu::_sfpu_reciprocal_<2>(m_plus_1);
            sfpi::vFloat z = m_minus_1 * m_plus_1_recip;

            // Compute z² for polynomial evaluation
            sfpi::vFloat z2 = z * z;

            // Polynomial approximation using odd powers
            // ln(m) = 2z(1 + z²/3 + z⁴/5 + z⁶/7 + z⁸/9 + z¹⁰/11)
            // Using Horner's method on z² with coefficients
            constexpr float c11 = 1.0f / 11.0f;
            constexpr float c9 = 1.0f / 9.0f;
            constexpr float c7 = 1.0f / 7.0f;
            constexpr float c5 = 0.2f;
            constexpr float c3 = 1.0f / 3.0f;

            sfpi::vFloat p = sfpi::PolynomialEvaluator::eval(
                z2,
                sfpi::vConst1,  // c0 = 1
                c3,             // c1 = 1/3
                c5,             // c2 = 1/5
                c7,             // c3 = 1/7
                c9,             // c4 = 1/9
                c11             // c5 = 1/11
            );

            // Final computation: ln(m) = 2 * z * p
            sfpi::vFloat ln_m = TWO * z * p;

            // Combine: ln(1+x) = exp×ln(2) + ln(m)
            // Convert exponent to float using sign-magnitude format
            sfpi::vInt exp_for_convert = exp;

            // Convert negative exponents to sign-magnitude format
            v_if(exp < 0) {
                sfpi::vInt exp_abs = ~exp + 1;               // Two's complement negation
                exp_for_convert = sfpi::setsgn(exp_abs, 1);  // Set sign bit for negative
            }
            v_endif;

            sfpi::vFloat expf = sfpi::int32_to_float(exp_for_convert, 0);

            result = expf * LN2 + ln_m;
        }
        v_endif;
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        // Convert to bfloat16 if needed using round-to-nearest-even
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log1p_init() {
    if constexpr (!is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm0 = 0.692871f;  // ln2
        sfpi::vConstFloatPrgm1 = 0.1058f;
        sfpi::vConstFloatPrgm2 = -0.7166f;
    } else {
        // Initialize reciprocal kernel for division operations in the ln computation path
        _init_reciprocal_</*approximation_mode*/ false, /*legacy_compat*/ false>();
    }
}

}  // namespace sfpu
}  // namespace ckernel

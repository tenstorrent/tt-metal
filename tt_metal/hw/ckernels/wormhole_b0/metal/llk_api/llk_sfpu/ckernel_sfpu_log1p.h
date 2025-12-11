// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_log.h"

namespace ckernel {
namespace sfpu {

/*
 * This function implements ln(1+x) using Chebyshev approximation for bfloat16.
 * Uses 3rd order Chebyshev polynomial approximation with range reduction.
 *
 * @tparam FAST_APPROX If true, skip NaN check for negative inputs
 * @tparam is_fp32_dest_acc_en If false, round result to bfloat16
 * @param val The input value x
 * @return Result of ln(1+val)
 */
template <bool FAST_APPROX, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log1p_bf16(sfpi::vFloat val) {
    // Read programmable constants

    sfpi::vFloat in = val + sfpi::vConst1;
    sfpi::vFloat result =
        calculate_log_body<FAST_APPROX, /*HAS_BASE_SCALING*/ false, /*is_fp32_dest_acc_en*/ true>(in, 0);

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

/*
 * This function implements ln(1+x) with accurate computation for small x.
 * Algorithm based on log1p_fp32 from operations.py:
 * 1. Handle special cases (x < -1, x == -1, infinity, NaN)
 * 2. For |x| < 0.3: Use 13-term Taylor series to avoid catastrophic cancellation
 * 3. For |x| >= 0.3: Use standard ln(1+x) computation
 *
 * The threshold of 0.3 and 13 terms work well enough to provide good accuracy
 * across the entire domain.
 *
 * @param val The input value (sfpi::vFloat vector), can be any floating point number > -1
 * @return sfpi::vFloat Result of ln(1+val)
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log1p_fp32(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    // Check for special cases
    sfpi::vInt exp = sfpi::exexp(val);  // Get debiased exponent

    v_if(sfpi::reinterpret<sfpi::vInt>(val) == 0x7F800000) {
        // If input is infinity, return infinity
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(exp == 128 || val < -1.f) {                   // NaN or negative input -> NaN
        result = std::numeric_limits<float>::quiet_NaN();  // returns nan
    }
    v_elseif(val == -1.f) {
        // x = -1 input -> -inf
        result = -std::numeric_limits<float>::infinity();
    }
    v_else {
        // Normal computation
        sfpi::vFloat abs_val = sfpi::abs(val);

        constexpr float THRESHOLD = 0.3f;
        v_if(abs_val < THRESHOLD) {
            // For |x| < 0.3, use 13-term Taylor series to avoid catastrophic cancellation
            // ln(1+x) = x - x²/2 + x³/3 - x⁴/4 + ... + x¹³/13
            // Using PolynomialEvaluator with coefficients in ascending order:
            // ln(1+x) = 0 + x*(1 + x*(-1/2 + x*(1/3 + x*(-1/4 + ...))))
            result = PolynomialEvaluator::eval(
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
            // The following is the same approximation as calculate_log_f32_body from ckernel_sfpu_log.h.
            // Ideally, we would like to call calculate_log_f32_body() directly.
            // However, doing so leads to 'register spilling errors' due to interaction with the
            // polynomial evaluation near 0.

            // For |x| >= 0.3, use standard ln(1+x) computation
            sfpi::vFloat one_plus_x = sfpi::vConst1 + val;

            // Extract exponent (debiased) from (1+x)
            exp = sfpi::exexp(one_plus_x);

            // Extract mantissa and construct m in [1, 2)
            // Use setexp to normalize to [1, 2) range by setting exponent to 127 (bias)
            sfpi::vFloat m = sfpi::setexp(one_plus_x, 127);

            // Step 2: Range reduction
            // If m >= sqrt(2), divide by 2 and increment exponent
            // This ensures m is in [sqrt(2)/2, sqrt(2)] ≈ [0.707, 1.414]
            constexpr float SQRT2 = 1.4142135381698608f;  // sqrt(2)
            v_if(m >= SQRT2) {
                // m = m * 0.5f;  // Divide by 2
                m = m * 0.5f;
                exp = exp + 1;  // Increment exponent
            }
            v_endif;

            // Transform to z = (m - 1) / (m + 1)
            // This maps m ∈ [0.707, 1.414] to z ∈ [-0.172, 0.172]
            // ln(m) = 2 × (z + z³/3 + z⁵/5 + z⁷/7 + ...)
            sfpi::vFloat m_minus_1 = m - sfpi::vConst1;
            sfpi::vFloat m_plus_1 = m + sfpi::vConst1;

            // Compute z = (m - 1) / (m + 1) using reciprocal
            // z = m_minus_1 * (1 / m_plus_1)
            sfpi::vFloat m_plus_1_recip = ckernel::sfpu::_sfpu_reciprocal_<2>(m_plus_1);
            sfpi::vFloat z = m_minus_1 * m_plus_1_recip;

            // Compute z**2 for polynomial evaluation
            sfpi::vFloat z2 = z * z;

            // Step 4: Polynomial approximation using odd powers
            // ln(m) = 2z(1 + (z**2)/3 + (z**4)/5 + (z**6)/7 + (z**8)/9 + (z**10)/11)
            // Using Horner's method: p = 1 + z**2 * (c3 + z**2 * (c5 + z**2 * (c7 + z**2 * (c9 + z**2 * c11))))

            // Polynomial coefficients for ln(m) where m in [sqrt(2)/2, sqrt(2)]
            // ln(m) = 2z(1 + z²/3 + z⁴/5 + z⁶/7 + z⁸/9 + z¹⁰/11)
            // where z = (m - 1) / (m + 1)
            sfpi::vFloat p = PolynomialEvaluator::eval(
                z2,
                sfpi::vConst1,
                0.3333333333333333f,
                0.2f,
                0.14285714285714285f,
                0.1111111111111111f,
                .09090909090909091f);

            // Final computation: ln(m) = 2 * z * p
            sfpi::vFloat ln_m = 2.f * z * p;

            // Combine: ln(1+x) = exp×ln(2) + ln(m)
            // Convert exponent to float using sign-magnitude format
            sfpi::vInt exp_for_convert = exp;

            // We want to convert exponent to floating point using int32 -> float conversion.
            // However, int32_to_float takes a sign-magnitude
            // This is not an issue for positive numbers (same representation)
            // For negative numbers, we need to explicitly convert to sign-magnitude format
            v_if(exp < 0) {
                sfpi::vInt exp_abs = ~exp + 1;  // Two's complement negation
                // Convert to sign-magnitude negative: setsgn(value, 1) sets MSB to 1
                exp_for_convert = sfpi::setsgn(exp_abs, 1);
            }
            v_endif;

            sfpi::vFloat expf = sfpi::int32_to_float(exp_for_convert, 0);

            // Step 5: Combine: ln(x) = exp×ln(2) + ln(m)
            constexpr float LN2 = 0.69314718246459961f;  // log(2)
            result = expf * LN2 + ln_m;                  // log(x) = log2(x) / log(2)
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

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_log1p() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;
        if constexpr (is_fp32_dest_acc_en) {
            result = calculate_log1p_fp32<is_fp32_dest_acc_en>(in);
        } else {
            result = calculate_log1p_bf16<FAST_APPROX, is_fp32_dest_acc_en>(in);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log1p_init() {
    if constexpr (!is_fp32_dest_acc_en) {
        log_init<APPROXIMATION_MODE, FAST_APPROX, is_fp32_dest_acc_en>();
    } else {
        _init_reciprocal_</*approximation_mode*/ false, /*legacy_compat*/ false>();
    }
}

}  // namespace sfpu
}  // namespace ckernel

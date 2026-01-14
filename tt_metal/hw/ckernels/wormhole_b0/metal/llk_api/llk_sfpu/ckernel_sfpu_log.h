// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel {
namespace sfpu {

template <bool FAST_APPROX, bool HAS_BASE_SCALING, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log_body(sfpi::vFloat in, const uint log_base_scale_factor) {
    ///////////////////////////////////
    // "normalize to calculation range"
    ///////////////////////////////////
    sfpi::vFloat x = sfpi::setexp(in, 127);  // set exp to exp bias (put in range of 1-2)

    // Minimax approximation of log(x) over [1; 2] calculated using Sollya with the following command:
    // > fpminimax(log(x), 5, [|single...|], [1+2^(-20); 2], relative);
    sfpi::vFloat series_result = PolynomialEvaluator::eval(
        x,
        sfpi::vConstFloatPrgm1,
        sfpi::vConstFloatPrgm2,
        -2.800232410430908,
        1.3681391477584839,
        -0.3706687390804291,
        0.04224011301994324);

    ////////////////////////////
    // Convert exponent to float
    ////////////////////////////
    sfpi::vInt exp = sfpi::exexp(in);

    // Convert negative numbers: signed -> sign-magnitude
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;

    sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat result = expf * vConstLn2 + series_result;  // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING) {
        result *= sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
    }

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    ////////////////////////////
    v_if(in == 0.0F) {  // Reload for register pressure
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    if constexpr (!FAST_APPROX) {
        sfpi::vInt exp = sfpi::exexp(in);
        v_if(sfpi::reinterpret<sfpi::vInt>(in) == 0x7F800000) {
            // If input is infinity, return infinity
            result = std::numeric_limits<float>::infinity();
        }
        v_elseif(exp == 128 || in < 0.f) {                     // +inf or negative input -> NaN
            result = std::numeric_limits<float>::quiet_NaN();  // returns nan for fp32 and inf for bf16
        }
        v_endif;
    }

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

/*
 * This function implements ln(x) using polynomial approximation.
 * 1. Handle special cases (x <= 0, infinity, NaN)
 * 2. Extract exponent and mantissa: x = 2^n × m
 * 3. Reduce range: adjust m to be in [sqrt(2)/2, sqrt(2)]
 * 4. Compute ln(m) using polynomial approximation
 * 5. Return n×ln(2) + ln(m)
 */
template <bool HAS_BASE_SCALING>
sfpi_inline sfpi::vFloat calculate_log_f32_body(sfpi::vFloat val, const uint log_base_scale_factor) {
    sfpi::vFloat result;

    // Check for special cases
    sfpi::vInt exp = sfpi::exexp(val);  // Get debiased exponent

    v_if(sfpi::reinterpret<sfpi::vInt>(val) == 0x7F800000) {
        // If input is infinity, return infinity
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(exp == 128 || val < 0.f) {                    // +inf or negative input -> NaN
        result = std::numeric_limits<float>::quiet_NaN();  // returns nan for fp32 and inf for bf16
    }
    v_elseif(val == 0.0f) {
        // Zero input -> -inf
        result = -std::numeric_limits<float>::infinity();
    }
    v_else {
        // Step 1: Extract exponent and mantissa
        // Extract mantissa and construct m in [1, 2)
        // Use setexp to normalize to [1, 2) range by setting exponent to 127 (bias)
        sfpi::vFloat m = sfpi::setexp(val, 127);

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

        // Step 3: Transform to z = (m - 1) / (m + 1)
        // This maps m ∈ [0.707, 1.414] to z ∈ [-0.172, 0.172]
        // ln(m) = 2 × (z + z³/3 + z⁵/5 + z⁷/7 + ...)
        sfpi::vFloat m_minus_1 = m - sfpi::vConst1;
        sfpi::vFloat m_plus_1 = m + sfpi::vConst1;

        // Compute z = (m - 1) / (m + 1) using reciprocal
        // z = m_minus_1 * (1 / m_plus_1)
        sfpi::vFloat m_plus_1_recip = _sfpu_reciprocal_<2>(m_plus_1);
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
        sfpi::vFloat ln_m = 2.0f * (z * p);

        // We want to convert exponent to floating point using int32 -> float conversion.
        // However, int32_to_float takes a sign-magnitude
        // This is not an issue for positive numbers (same representation)
        // For negative numbers, we need to explicitly convert to sign-magnitude format
        v_if(exp < 0) {
            // Compute absolute value: ~exp + 1 (two's complement negation)
            sfpi::vInt exp_abs = ~exp + 1;
            // Convert to sign-magnitude negative: setsgn(value, 1) sets MSB to 1
            exp = sfpi::setsgn(exp_abs, 1);
        }
        v_endif;

        // Convert to float - int32_to_float handles sign-magnitude format correctly
        sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);

        // Step 5: Combine: ln(x) = exp×ln(2) + ln(m)
        constexpr float LN2 = 0.69314718246459961f;  // log(2)
        result = expf * LN2 + ln_m;                 // log(x) = log2(x) / log(2)

        if constexpr (HAS_BASE_SCALING) {
            result *= sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
        }
    }
    v_endif;

    return result;
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool HAS_BASE_SCALING,
    bool is_fp32_dest_acc_en,
    int ITERATIONS = 8>
inline void calculate_log(uint log_base_scale_factor) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;
        if constexpr (!is_fp32_dest_acc_en) {
            result = calculate_log_body<FAST_APPROX, HAS_BASE_SCALING, is_fp32_dest_acc_en>(in, log_base_scale_factor);
        } else {
            result = calculate_log_f32_body<HAS_BASE_SCALING>(in, log_base_scale_factor);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log_init() {
    if constexpr (!is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm0 = 0.69314718246459961f;  // ln(2)
        sfpi::vConstFloatPrgm1 = -2.0069785118103027;
        sfpi::vConstFloatPrgm2 = 3.767500400543213;
    } else {
        _init_reciprocal_</*approximation_mode*/ false, /*legacy_compat*/ false>();
    }
}

}  // namespace sfpu
}  // namespace ckernel

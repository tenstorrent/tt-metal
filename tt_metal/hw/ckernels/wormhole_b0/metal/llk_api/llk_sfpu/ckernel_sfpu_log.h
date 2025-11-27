// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_polyval.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool FAST_APPROX, bool HAS_BASE_SCALING, bool is_fp32_dest_acc_en = false>
sfpi_inline void calculate_log_body(sfpi::vFloat in, const uint log_base_scale_factor) {
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
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;

    sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat result = expf * vConstLn2 + series_result;  // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING) {
        result *= sfpi::s2vFloat16a(log_base_scale_factor);
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
        sfpi::vInt man = sfpi::exman9(in);
        sfpi::vInt signbit = sfpi::reinterpret<sfpi::vInt>(in) & 0x80000000;  // returns 0 for +ve value
        v_if((exp == 128 && man != 0) || in < 0.0F) {
            result = std::numeric_limits<float>::quiet_NaN();  // returns nan for fp32 and inf for bf16
        }
        v_elseif(signbit == 0 && exp == 128 && man == 0) { result = std::numeric_limits<float>::infinity(); }
        v_endif;
    }

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }
}

/*
 * This function implements ln(x) using polynomial approximation.
 * Algorithm based on ln_fp32 from operations.py:
 * 1. Handle special cases (x <= 0, infinity, NaN)
 * 2. Extract exponent and mantissa: x = 2^n × m
 * 3. Reduce range: adjust m to be in [sqrt(2)/2, sqrt(2)]
 * 4. Compute ln(m) using polynomial approximation
 * 5. Return n×ln(2) + ln(m)
 *
 * @param val The input value (sfpi::vFloat vector), can be any floating point number
 * @return sfpi::vFloat Result of ln(val)
 */
sfpi_inline sfpi::vFloat calculate_log_f32_body(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    // Constants (all in fp32)
    constexpr float LN2 = 0.6931471805599453f;    // ln(2)
    constexpr float SQRT2 = 1.4142135623730951f;  // sqrt(2)
    constexpr float HALF = 0.5f;
    constexpr float ONE = 1.0f;
    constexpr float TWO = 2.0f;
    constexpr float ZERO = 0.0f;
    constexpr int BIAS = 127;  // IEEE 754 bias

    // Polynomial coefficients for ln(m) where m in [sqrt(2)/2, sqrt(2)]
    // ln(m) = 2z(1 + z²/3 + z⁴/5 + z⁶/7 + z⁸/9 + z¹⁰/11)
    // where z = (m - 1) / (m + 1)
    constexpr float c11 = 0.09090909090909091f;  // 1/11
    constexpr float c9 = 0.1111111111111111f;    // 1/9
    constexpr float c7 = 0.14285714285714285f;   // 1/7
    constexpr float c5 = 0.2f;                   // 1/5
    constexpr float c3 = 0.3333333333333333f;    // 1/3

    // Check for special cases
    sfpi::vInt exp_bits = sfpi::exexp(val);   // Get debiased exponent
    sfpi::vInt man_bits = sfpi::exman9(val);  // Get mantissa bits

    // Check for NaN: exponent = 128 (255 - 127, meaning original exp = 255) and mantissa != 0
    // Actually, exexp returns debiased, so exp_bits == 128 means original exp = 255
    v_if(exp_bits == 128 && man_bits != 0) {
        // NaN: exponent = 255 and mantissa != 0
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_elseif(val < sfpi::vFloat(ZERO)) {
        // Negative input
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_elseif(val == sfpi::vFloat(ZERO)) {
        // Zero input
        result = -std::numeric_limits<float>::infinity();
    }
    v_elseif(exp_bits == 128 && man_bits == 0) {
        // Infinity: exponent = 255 and mantissa = 0
        result = std::numeric_limits<float>::infinity();
    }
    v_else {
        // Step 1: Extract exponent and mantissa
        // Extract exponent (debiased) - exexp already returns debiased exponent
        sfpi::vInt exp = sfpi::exexp(val);

        // Extract mantissa and construct m in [1, 2)
        // Use setexp to normalize to [1, 2) range by setting exponent to 127 (bias)
        sfpi::vFloat m = sfpi::setexp(val, 127);

        // Step 2: Range reduction
        // If m >= sqrt(2), divide by 2 and increment exponent
        // This ensures m is in [sqrt(2)/2, sqrt(2)] ≈ [0.707, 1.414]
        v_if(m >= sfpi::vFloat(SQRT2)) {
            m = m * sfpi::vFloat(HALF);  // Divide by 2
            exp = exp + 1;               // Increment exponent
        }
        v_endif;

        // Step 3: Transform to z = (m - 1) / (m + 1)
        // This maps m ∈ [0.707, 1.414] to z ∈ [-0.172, 0.172]
        // ln(m) = 2 × (z + z³/3 + z⁵/5 + z⁷/7 + ...)
        sfpi::vFloat m_minus_1 = m - sfpi::vFloat(ONE);
        sfpi::vFloat m_plus_1 = m + sfpi::vFloat(ONE);

        // Compute z = (m - 1) / (m + 1) using reciprocal
        // z = m_minus_1 * (1 / m_plus_1)
        sfpi::vFloat m_plus_1_recip = ckernel::sfpu::_sfpu_reciprocal_<2>(m_plus_1);
        sfpi::vFloat z = m_minus_1 * m_plus_1_recip;

        // Compute z² for polynomial evaluation
        sfpi::vFloat z2 = z * z;

        // Step 4: Polynomial approximation using odd powers
        // ln(m) = 2z(1 + z²/3 + z⁴/5 + z⁶/7 + z⁸/9 + z¹⁰/11)
        // Using Horner's method: p = 1 + z²*(c3 + z²*(c5 + z²*(c7 + z²*(c9 + z²*c11))))
        sfpi::vFloat p = sfpi::vFloat(c11);

        sfpi::vFloat temp = z2 * p;
        p = sfpi::vFloat(c9) + temp;

        temp = z2 * p;
        p = sfpi::vFloat(c7) + temp;

        temp = z2 * p;
        p = sfpi::vFloat(c5) + temp;

        temp = z2 * p;
        p = sfpi::vFloat(c3) + temp;

        temp = z2 * p;
        p = sfpi::vFloat(ONE) + temp;

        // Final computation: ln(m) = 2 * z * p
        temp = z * p;
        sfpi::vFloat ln_m = sfpi::vFloat(TWO) * temp;

        // Step 5: Combine: ln(x) = exp×ln(2) + ln(m)
        // Convert exponent to float (exp is already debiased from exexp)
        // Follow the same pattern as ckernel_sfpu_log.h:
        // For negative exp, convert to sign-magnitude format for int32_to_float
        // int32_to_float automatically interprets the sign bit correctly
        sfpi::vInt exp_for_convert = exp;

        // Convert negative exponents to sign-magnitude format
        // setsgn(abs_value, 1) creates negative in sign-magnitude (MSB=1)
        // int32_to_float will interpret this as negative float
        v_if(exp < 0) {
            // Compute absolute value: ~exp + 1 (two's complement negation)
            sfpi::vInt exp_abs = ~exp + 1;
            // Convert to sign-magnitude negative: setsgn(value, 1) sets MSB to 1
            exp_for_convert = sfpi::setsgn(exp_abs, 1);
        }
        v_endif;

        // Convert to float - int32_to_float handles sign-magnitude format correctly
        // For sign-magnitude: MSB=1 means negative, MSB=0 means positive
        sfpi::vFloat expf = sfpi::int32_to_float(exp_for_convert, 0);

        temp = expf * sfpi::vFloat(LN2);
        result = temp + ln_m;
    }
    v_endif;

    //  if constexpr (!is_fp32_dest_acc_en) {
    //      // Convert to bfloat16 if needed
    //      result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    //  }

    return result;
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool HAS_BASE_SCALING,
    bool is_fp32_dest_acc_en = false,
    int ITERATIONS = 8>
inline void calculate_log(uint log_base_scale_factor) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;
        if constexpr (!is_fp32_dest_acc_en) {
            result = calculate_log_body<FAST_APPROX, HAS_BASE_SCALING, is_fp32_dest_acc_en>(in, log_base_scale_factor);
        } else {
            result = calculate_log_f32_body(in);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en = false>
inline void log_init() {
    if constexpr (!is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm0 = 0.693147182464599609375;  // ln(2)

        // XXXXX could do these to higher precision
        sfpi::vConstFloatPrgm1 = -2.0069785118103027;
        sfpi::vConstFloatPrgm2 = 3.767500400543213;
    } else {
        _init_reciprocal_</*approximation_mode*/ false, /*legacy_compat*/ false>();
    }
}

}  // namespace sfpu
}  // namespace ckernel

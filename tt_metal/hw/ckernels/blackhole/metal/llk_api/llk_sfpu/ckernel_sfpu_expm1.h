// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel {
namespace sfpu {

/*
 * This function implements expm1(x) = exp(x) - 1 using a polynomial approximation algorithm
 * based on "Simple Multiple Precision Algorithms for Exponential Functions [Tips & Tricks]"
 * by Moroz et al. 2022 (https://doi.org/10.1109/MSP.2022.3157460).
 * More specifically, it is the implementation of the `exp_21f` algorithm described in Section 5
 *
 * @param val The input value (sfpi::vFloat vector), can be any floating point number
 *
 * @return sfpi::vFloat Result of expm1(val)
 *
 * @see Moroz et al. 2022 - "Simple Multiple Precision Algorithms for Exponential Functions"
 *      ( https://doi.org/10.1109/MSP.2022.3157460 )
 */
template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_expm1_(sfpi::vFloat val) {
    sfpi::vFloat y = sfpi::vConstNeg1;
    v_if(sfpi::abs(val) < sfpi::s2vFloat16b(0.4f)) {
        // When x is very small, exp(x) is very close to 1. Hence, for improved precision, we use Taylor expansion of
        // expm1(x) = x + (x^2/2) + (x^3/6)
        // In Horner form, on reducing further : y = (val * (val * (val * 0.166f + 0.5f )+ 1)
        y = val * (sfpi::vConst1 + val * (sfpi::vFloat(0.5f) + val * sfpi::vFloat(0.166f)));
    }
    v_elseif(val > sfpi::vFloat(-88.0f)) {
        // The paper relies on the following formula (c.f. Section 2 and 3 of paper):
        // z = (bias + x * factor * N_m; where:
        // factor = 0x00b8aa3b (computed through log(e))
        // bias = 0x3f800000
        sfpi::vInt z = sfpu::_float_to_int32_exp21f_(val * sfpi::vFloat(0x00b8aa3b) + sfpi::vFloat(0x3f800000));
        sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));
        sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));

        sfpi::vFloat d1 = sfpi::vFloat(sfpi::vConstFloatPrgm0);
        sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vConstIntPrgm1 + zif, 0);
        sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vConstIntPrgm2 + zif, 0);
        d2 = d1 * d2;
        zif = sfpu::_float_to_int32_exp21f_(d2 * d3);

        // Restore exponent
        zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));

        y = sfpi::reinterpret<sfpi::vFloat>(zii) - sfpi::vConst1;
    }
    v_endif;
    if constexpr (!is_fp32_dest_acc_en) {
        // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
        // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
        // rather than 81 (which would have been correct).
        // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest-even.
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }
    return y;
}

/*
 * This function implements expm1(x) = exp(x) - 1 with high accuracy for float32.
 * Target accuracy: < 2 ULP for float32.
 *
 * Uses hybrid approach optimized for maximum ULP error:
 * - Taylor series (order 8) for |x| < 0.5 to avoid catastrophic cancellation
 * - exp(x) - 1 for |x| >= 0.5 (calls _sfpu_exp_f32_accurate_)
 *
 * This avoids catastrophic cancellation when x is near 0.
 *
 * @param val The input value (sfpi::vFloat vector), can be any floating point number
 * @return sfpi::vFloat Result of expm1(val)
 */
sfpi_inline sfpi::vFloat _sfpu_expm1_f32_accurate_(sfpi::vFloat val) {
    // Start with default result for large negative values (expm1 -> -1)
    sfpi::vFloat result = sfpi::vConstNeg1;

    // Constants
    constexpr float THRESHOLD = 0.5f;
    // Overflow threshold: log(2) * 128 = 88.72
    constexpr float OVERFLOW_THRESHOLD = 88.72283935546875f;

    sfpi::vFloat abs_val = sfpi::abs(val);

    // For small |x| < 0.5: use Taylor series to avoid cancellation
    v_if(abs_val < THRESHOLD) {
        // expm1(x) = x + x^2/2! + x^3/3! + ... + x^8/8!
        // Using PolynomialEvaluator for Horner's method
        result = PolynomialEvaluator::eval(
            val,
            sfpi::vConst0,   // c0 = 0
            sfpi::vConst1,   // c1 = 1/1! = 1
            0.5f,            // c2 = 1/2!
            1.0f / 6.0f,     // c3 = 1/3!
            1.0f / 24.0f,    // c4 = 1/4!
            1.0f / 120.0f,   // c5 = 1/5!
            1.0f / 720.0f,   // c6 = 1/6!
            1.0f / 5040.0f,  // c7 = 1/7!
            1.0f / 40320.0f  // c8 = 1/8!
        );
    }
    v_elseif(val >= OVERFLOW_THRESHOLD) {
        // Overflow: expm1(large positive) = +infinity
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(val > sfpi::vFloat(-88.0f)) {
        // For moderate values: use exp(x) - 1
        // This is accurate because exp(x) is not close to 1
        // Call the accurate exp implementation and subtract 1
        sfpi::vFloat exp_result = _sfpu_exp_f32_accurate_(val);
        result = exp_result - sfpi::vConst1;
    }
    v_endif;
    // For val <= -88, result stays at -1 (correct for expm1)

    return result;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_expm1_improved_(sfpi::vFloat val);

// is_fp32_dest_acc_en == false: use bfloat16-optimized version
template <>
sfpi_inline sfpi::vFloat _sfpu_expm1_improved_<false>(sfpi::vFloat val) {
    return _sfpu_expm1_<false>(val);
}

// is_fp32_dest_acc_en == true: use float32-accurate version
template <>
sfpi_inline sfpi::vFloat _sfpu_expm1_improved_<true>(sfpi::vFloat val) {
    return _sfpu_expm1_f32_accurate_(val);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_expm1() {
    if constexpr (APPROXIMATION_MODE) {
        // Use original approximation mode
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat v = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _sfpu_expm1_<is_fp32_dest_acc_en>(v);
            sfpi::dst_reg++;
        }
    } else {
        // Use improved version based on destination precision
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat v = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _sfpu_expm1_improved_<is_fp32_dest_acc_en>(v);
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void expm1_init() {
    if constexpr (APPROXIMATION_MODE || !is_fp32_dest_acc_en) {
        // Polynomial coefficients for approximation of exp on [1; 2]
        // Used by the approximation mode and bfloat16 mode
        sfpi::vConstFloatPrgm0 = 0.40196114e-7f;
        sfpi::vConstIntPrgm1 = 0xf94ee7;
        sfpi::vConstIntPrgm2 = 0x560e;
    }
    // For accurate float32 mode (!APPROXIMATION_MODE && is_fp32_dest_acc_en),
    // no programmable constants are needed as the implementation uses inline constants
}

}  // namespace sfpu
}  // namespace ckernel

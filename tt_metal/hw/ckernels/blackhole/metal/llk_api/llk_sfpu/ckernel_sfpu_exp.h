// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat sfpu_exp(sfpi::vFloat val) { return _sfpu_exp_(val); }

/*
 * Both _float_to_int32_ and _float_to_int32_positive_ use branch to handle special cases
 * With exp21f function, some of these cases never happen (e.g. negative exponent, overflow)
 * This allow for a branch free (and much smaller algorithm) to compute integer value
 *
 * The constraint on `val` is: 0 <= val < 128.0f
 * Note: Unlike _float_to_int32_ and _float_to_int32_positive, this function assumes that
 * value has been been divided by 2^23. Output value will be scaled by 2^23 compared to 'val'.
 * If that was not the case, we would have had to shift by `exp - 23` instead of `exp`
 * This saves 1 SFPADDI instruction.
 */
sfpi_inline sfpi::vInt _float_to_int32_for_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);
    sfpi::vInt man = sfpi::exman8(val);  // get mantissa with implicit bit (man in [1; 2])
    man = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));
    return man;
}

/*
 * This function implements the exponential function using a polynomial approximation algorithm
 * based on "Simple Multiple Precision Algorithms for Exponential Functions [Tips & Tricks]"
 * by Moroz et al. 2022 (https://doi.org/10.1109/MSP.2022.3157460).
 * More specifically, it is the implementation of the `exp_21f` algorithm described in Section 5
 *
 * @param val The input value (sfpi::vFloat vector), can be any floating point number
 *
 * @return sfpi::vFloat Result of exp(val)
 *
 * @see Moroz et al. 2022 - "Simple Multiple Precision Algorithms for Exponential Functions"
 *      ( https://doi.org/10.1109/MSP.2022.3157460 )
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    // This function computes exp(x) by leveraging mathematic properties of exp(x):
    // That is, exp(x) = 2**(x / ln2) = 2**(x_i) * 2**(x_f) where
    // - z_i = trunc(x / ln2) (integer part)
    // - z_f = x/ln2 - trunc(x/ln2) (fractional part)
    //
    // The paper relies on the following formula (c.f. Section 2 and 3 of paper):
    // z = (bias + x * factor * N_m); where:
    // factor = log(2) * 2^23
    // bias = 127 * 2^23
    // Fundamentally, the formula in the paper computes
    // z = val * log(2) * 2^23 + 127 * 2^23
    // This formula prepares for the computation of exp(x) = 2^(x/log(2))
    //
    // In our case, we will let the multiplication by 2^23 be done implicitly in _float_to_int32_exp21f_ function
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);

    // Intermediary values can overflow in xlog2 is outside of [0, 256[ which leads to invalid results instead of 0
    // (when input < -88.5) and +inf (when input > 88.5)
    // To avoid this, we clamp xlog2 to [0, 255]
    // (thresholds values are rounded to bf16, as it does not change result but only requires one SFPLOADI vs. two)
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);
    sfpi::vec_min_max(xlog2, threshold_high);

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    sfpi::vInt exponential_part =
        exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract exponent ( = 2**(integer part of val/ln2))
    sfpi::vInt fractional_part =
        sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract mantissa ( = leftover part, in [0; 1])

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);

    // To refine approximation of 2**(x_f), we use an approximation of 2**x on [0; 2^23]
    // This uses a 2nd degree polynomial adjustment of the fractional part
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombined exponent and mantissa: this is equivalent to 2**(x_i) * 2**(x_f)
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

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
 * This function implements the exponential function using a polynomial approximation algorithm
 * based on "Simple Multiple Precision Algorithms for Exponential Functions [Tips & Tricks]"
 * by Moroz et al. 2022 (https://doi.org/10.1109/MSP.2022.3157460).
 * More specifically, it is the implementation of the `exp_61f` algorithm described in Section 5
 *
 * @param val The input value (sfpi::vFloat vector), can be any floating point number
 *
 * @return sfpi::vFloat Result of exp(val)
 *
 * @see Moroz et al. 2022 - "Simple Multiple Precision Algorithms for Exponential Functions"
 *      ( https://doi.org/10.1109/MSP.2022.3157460 )
 */
sfpi_inline sfpi::vFloat _sfpu_exp_61f_(sfpi::vFloat val) {
    // This function computes exp(x) by leveraging mathematic properties of exp(x):
    // That is, exp(x) = 2**(x / ln2) = 2**(x_i) * 2**(x_f) where
    // - z_i = trunc(x / ln2) (integer part)
    // - z_f = x/ln2 - trunc(x/ln2) (fractional part)
    //
    // The paper relies on the following formula (c.f. Section 2 and 3 of paper):
    // z = (bias + x * factor * N_m); where:
    // factor = log(2) * 2^23
    // bias = 127 * 2^23
    // Fundamentally, the formula in the paper computes
    // z = val * log(2) * 2^23 + 127 * 2^23
    // This formula prepares for the computation of exp(x) = 2^(x/log(2))
    //
    // In our case, we will let the multiplication by 2^23 be done implicitly in _float_to_int32_exp21f_ function
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = val * ONE_LN2 + 127.f;

    // Intermediary values can overflow in xlog2 is outside of [0, 256[ which leads to invalid results instead of 0
    // (when input < -88.5) and +inf (when input > 88.5)
    // To avoid this, we clamp xlog2 to [0, 255]
    // (thresholds values are rounded to bf16, as it does not change result but only requires one SFPLOADI vs. two)
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);
    sfpi::vec_min_max(xlog2, threshold_high);

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    sfpi::vInt exponential_part =
        exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract exponent ( = 2**(integer part of val/ln2))
    sfpi::vInt fractional_part =
        sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract mantissa ( = leftover part, in [0; 1])

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);
    // Multiply by 2^-23
    // We could have scaled polynomial coefficients, but the last one would have been near the subnormal range (i.e.
    // truncation risk)
    frac = sfpi::addexp(frac, -23);

    // To refine approximation of 2**(x_f), we use an approximation of 2**x on [0; 1]
    // This uses a 2nd degree polynomial adjustment of the fractional part
    frac = PolynomialEvaluator::eval(
        frac, sfpi::vConst1, 0.69314699f, 0.24022982f, 0.055483369f, 0.0096788315f, 0.001243946f, 0.0002170391f);

    // Recombined exponent and mantissa: this is equivalent to 2**(x_i) * 2**(x_f)
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    return y;
}

// Utility function to round a float to a 32-bit integer while also calculating the
// integer part of the rounded value
sfpi_inline sfpi::vFloat _sfpu_round_nearest_int32_(sfpi::vFloat z, sfpi::vInt& k_int) {
    // From Hacker's Delight: round-to-nearest-even method
    // float -> int32 (round to nearest even): n = (x + float(c231)) - int32(c231)
    // round-to-nearest-even: n = (x + float(c231)) - float(c231)
    // where c231 = 0x4B400000 (2^23 + 2^22)
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);  // 2^23 + 2^22

    sfpi::vFloat tmp = z + c231;
    sfpi::vFloat k = tmp - c231;
    k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);

    return k;
}

/*
 * This function implements exp(x) using Cody-Waite range reduction for improved accuracy.
 * Target accuracy: < 1 ULP for float32.
 *
 * Algorithm:
 * 1. Handle special cases (overflow, underflow, NaN)
 * 2. Convert to base-2: exp(x) = 2^(x/ln2)
 * 3. Range reduction using Cody-Waite: compute k, then r = x - k*ln2_hi - k*ln2_lo
 * 4. Compute exp(r) using polynomial approximation (Taylor series)
 * 5. Scale by 2^k: result = 2^k * exp(r)
 *
 * @param val The input value (sfpi::vFloat vector), can be any floating point number
 * @return sfpi::vFloat Result of exp(val)
 */
sfpi_inline sfpi::vFloat _sfpu_exp_f32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    // Exp computation uses bit-wise manipulation using exponent and mantissa fields
    // For large values (e.g. |x| > 89), some intermediate values can overflow
    // To avoid this, we check the value of the input using two thresholds.
    //
    // These thresholds are applied after scaling x by 1/log(2) (i.e., on z = x * 1/ln(2)).
    // Mapped back to the original x domain, they correspond to approximately -88 and 89.
    constexpr float OVERFLOW_THRESHOLD = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;

    // Step 1: Compute k = round(x / ln(2))
    // z = x / ln(2) = x * (1/ln(2))
    constexpr float INV_LN2 = 1.4426950408889634f;  // 1/ln(2)
    sfpi::vFloat z = val * INV_LN2;

    // Check for special cases
    sfpi::vInt exp_bits = sfpi::exexp(z);

    v_if(z >= OVERFLOW_THRESHOLD) {
        // Overflow
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(z <= UNDERFLOW_THRESHOLD) {
        // Underflow
        result = sfpi::vConst0;
    }
    v_elseif(exp_bits == 255) {
        // infinity (exp = 255 && man != 0) already taken care of by previous conditionals:
        // if input is infinity or -infinity, then either z >= OVERFLOW_THRESHOLD or z <= UNDERFLOW_THRESHOLD
        // would have been true and their cases have already been handled.
        // Thus, we know that if exp == 0 here, then man != 0 as well.
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_else {
        // Round z to nearest integer using round-to-nearest-even
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);

        // Step 2: Cody-Waite range reduction
        // Compute r = x - k*ln(2) in extended precision
        // r = x - k*LN2_HI - k*LN2_LO
        // This provides better accuracy than simple r = x - k*ln(2)
        // Cody-Waite constants: ln(2) split into high and low parts for extended precision.
        // LN2_HI is chosen so that k*LN2_HI can be computed exactly for integer k in the valid range.
        // LN2_LO contains the remainder: LN2_HI + LN2_LO ≈ -ln(2)

        // We want to do:
        // 1) r_hi = val - k * LN2_HI
        // 2) r = r_hi - k * LN2_LO
        // On Wormhole, we can only do VD = VA * VB + VC, so we need to transform the expressions to
        // ensure optimization to a single SFPMAD instruction.
        // On Blackhole, SFFPMAD has SFPMAD_MOD1_NEGATE_VA and SFPMAD_MOD1_NEGATE_VC for this purpose.
        // However, negating constants maintains consistency with Wormhole, and ensures higher chance
        // of optimization to a single SFPMAD instruction.
        // The transformation is as follows:
        // 1) r_hi = val + k * (-LN2_HI)
        // 2) r = r_hi + k * (-LN2_LO)
        // Where LN2_HI and LN2_LO are negated.
        // This way, compiler can more easily optimize this expression to a single SFPMAD instruction.
        constexpr float LN2_HI = -0.6931152343750000f;  // High bits of ln(2)
        constexpr float LN2_LO = -3.19461832987e-05f;   // Low bits of ln(2)

        // First subtract k * LN2_HI
        sfpi::vFloat r_hi = k * LN2_HI + val;

        // Then subtract k * LN2_LO
        sfpi::vFloat r = k * LN2_LO + r_hi;

        // Step 3: Polynomial approximation for exp(r) using Taylor series
        // exp(r) ~= 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5! + r⁶/6! + r⁷/7!
        // Use 7th order polynomial (Taylor series coefficients) for < 1 ULP accuracy
        // Coefficients in ascending order of powers: c0, c1, c2, c3, c4, c5, c6, c7
        sfpi::vFloat p = PolynomialEvaluator::eval(
            r,
            sfpi::vConst1,  // c0 = 1
            sfpi::vConst1,  // c1 = 1
            0.5f,           // c2 = 1/2!
            1.0f / 6.0f,    // c3 = 1/3!
            1.0f / 24.0f,   // c4 = 1/4!
            1.0f / 120.0f,  // c5 = 1/5!
            1.0f / 720.0f,  // c6 = 1/6!
            1.0f / 5040.0f  // c7 = 1/7!
        );

        // Step 4: Scale by 2^k using exponent manipulation
        // ldexp(p, k_int) = p * 2^k
        // We do this by adding k_int to the exponent of p
        // Get the current exponent of p (without bias)
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);
        // Add k_int to get the new exponent
        sfpi::vInt new_exp = p_exp + k_int;

        // Set the new exponent
        result = sfpi::setexp(p, new_exp);
    }
    v_endif;

    return result;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_(sfpi::vFloat val);

// is_fp32_dest_acc_en == false
template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<false>(sfpi::vFloat val) {
    return _sfpu_exp_21f_<false>(val);
}

// is_fp32_dest_acc_en == true
template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<true>(sfpi::vFloat val) {
    return _sfpu_exp_f32_accurate_(val);
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool SKIP_POSITIVE_CHECK = false>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_exponential_<APPROXIMATION_MODE, SCALE_EN, ITERATIONS, FAST_APPROX, SKIP_POSITIVE_CHECK>(
            exp_base_scale_factor);
    } else {
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);
            }
            sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale>();
}

}  // namespace sfpu
}  // namespace ckernel

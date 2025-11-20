// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"
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
 * Note: Unlike _float_to_int32_ and _float_to_int32_positive, this function assumes that
 * value has been been divide by 2^23
 * If that was not the case, we would have had to shift by `exp - 23` instead of `exp`
 * This saves 1 SFPADDI instruction.
 */
sfpi_inline sfpi::vInt _float_to_int32_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = exexp(val);
    sfpi::vInt man = exman8(val);  // get mantissa with implicit bit (man in [1; 2])
    man = sfpi::reinterpret<sfpi::vInt>(shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));
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
template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    // This function computes exp(x) by leverage mathematic properties of exp(x):
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
    sfpi::vFloat xlog2 = val * sfpi::vConstFloatPrgm0 + 127.f;

    // Intermedirary values can overflow in xlog2 is outside of [0, 256[ which leads to invalid resutls instead of 0
    // (when input < -88.5) and +inf (when input > 88.5)
    // To avoid this, we clamp xlog2 to [0, 255]
    // (thresholds values are rounded to bf16, as it does not change result but only requires one SFPLOADI vs. two)
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    vec_min_max(threshold_low, xlog2);
    vec_min_max(xlog2, threshold_high);

    sfpi::vInt z = _float_to_int32_exp21f_(xlog2);

    sfpi::vInt exponential_part =
        exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract exponent ( = 2**(integer part of val/ln2))
    sfpi::vInt fractional_part =
        sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract mantissa ( = leftover part, in [0; 1])

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);

    // To refine approximation of 2**(x_f), we use an approximation of 2**x on [0; 1]
    // This uses a 2nd degree polynomial adjustment of the fractional part
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombined exponent and mantissa: this is equivalent to 2**(x_i) * 2**(x_f)
    exponential_part = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(frac, exponential_part));  // restore exponent

    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(exponential_part);

    if constexpr (!is_fp32_dest_acc_en) {
        // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
        // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
        // rather than 81 (which would have been correct).
        // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest-even.
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

sfpi_inline sfpi::vFloat _sfpu_exp_61f_(sfpi::vFloat val) {
    sfpi::vFloat y = sfpi::vConst0;
    v_if(val > -87.3f) {
        // The paper relies on the following formula (c.f. Section 2 and 3 of paper):
        // z = (bias + x * factor * N_m); where:
        // factor = 0x00b8aa3b (computed through log(e))
        // bias = 0x3f800000
        sfpi::vInt z = sfpu::_float_to_int32_(val * sfpi::vFloat(0x00b8aa3b) + sfpi::vFloat(0x3f800000));
        sfpi::vInt zii = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z));   // Extract exponent
        sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract mantissa

        // Normalize mantissa field into a fractional value in [0,1)
        sfpi::vFloat frac = sfpi::int32_to_float(zif, 0) * sfpi::vFloat(1.1920929e-7f);

        // Evaluate degree-6 polynomial coefficients using Horner’s rule
        // Note: Unlike exp_21f, in exp_61f all polynomial coefficients are floating-point values.
        // In exp_21f, the paper mixes integer and float constants to perform bit-level manipulation of the exponent and
        // mantissa fields (using bit manipulation techniques - BMT) for exactness. In exp_61f, all coefficients are
        // floating-point values derived from the Chebyshev polynomial approach, making the implementation simpler and
        // purely mathematical without integer-based operations.
        sfpi::vFloat poly = POLYVAL7<sfpi::vFloat>(
            0.0002170391f, 0.001243946f, 0.0096788315f, 0.055483369f, 0.24022982f, 0.69314699f, 1.0000000018f, frac);

        // Restore exponent
        zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(poly, 127U + zii));
        y = sfpi::reinterpret<sfpi::vFloat>(zii);
    }
    v_endif;

    return y;
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
    return _sfpu_exp_61f_(val);
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool SKIP_POSITIVE_CHECK = false,
    bool is_fp32_dest_acc_en = false>
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
    if constexpr (APPROXIMATION_MODE) {
        _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale>();
    } else {
        sfpi::vConstFloatPrgm0 = 1.4426950216293334961f;
    }
}

}  // namespace sfpu
}  // namespace ckernel

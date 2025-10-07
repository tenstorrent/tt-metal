// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat sfpu_exp(sfpi::vFloat val) { return _sfpu_exp_(val); }

#define POLYVAL7(coef6, coef5, coef4, coef3, coef2, coef1, coef0, t4) \
    (t4 * (t4 * (t4 * (t4 * (t4 * (coef6 * t4 + coef5) + coef4) + coef3) + coef2) + coef1) + coef0)

/*
 * Both _float_to_int32_ and _float_to_int32_positive_ use branch to handle special cases
 * With exp21f function, some of these cases never happen (e.g. negative exponent, overflow)
 * This allow for a branch free (and much smaller algorithm) to compute integer value
 */
sfpi_inline sfpi::vInt _float_to_int32_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = exexp(val);
    sfpi::vInt man = exman8(val);  // get mantissa with implicit bit (man in [1; 2])
    sfpi::vInt shift = exp - 23;
    man = sfpi::reinterpret<sfpi::vInt>(shft(sfpi::reinterpret<sfpi::vUInt>(man), shift));
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
    sfpi::vFloat y = sfpi::vConst0;
    if constexpr (!is_fp32_dest_acc_en) {
        // Intermediary values can overflow if abs(val) is above 88.5f, which leads to output increasing again instead
        // of staying at 0 (or becoming finite on large inputs). This overflow happens when `| log2(e) * val | > 127.0f`,
        // which correspond to `|val| > 88.5f`
        // Intermediary values can overflow if values exceeds 88.72283935546875 or -88.72283172607421875
        // To prevent this, we clamp -88.5 < x < 89
        // (thresholds values are rounded to bf16, as it does not change result but only requires one SFPLOADI vs. two)
        sfpi::vFloat threshold_high = sfpi::vFloat(89);
        sfpi::vFloat threshold_low = sfpi::vFloat(-88.5);
        vec_min_max(threshold_low, val);
        vec_min_max(val, threshold_high);

        // The paper relies on the following formula (c.f. Section 2 and 3 of paper):
        // z = (bias + x * factor * N_m; where:
        // factor = 0x00b8aa3b (computed through log(e))
        // bias = 0x3f800000
        //
        // Fundamentally, this computes exp(x) = 2**(x / ln2) = 2**(x_i) * 2**(x_f) where
        // - z_i = trunc(x / ln2) (integer part)
        // - z_f = x/ln2 - trunc(x/ln2) (fractional part)
        sfpi::vInt z = _float_to_int32_exp21f_(val * sfpi::vFloat(0x00b8aa3b) + sfpi::vFloat(0x3f800000));
        sfpi::vInt exponential_part =
            exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract exponent ( = 2**(integer part of val/ln2))
        sfpi::vInt fractional_part =
            sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract mantissa ( = leftover part, in [0; 1])

        // To refine approximation of 2**(x_f), we use an approximation of 2**x on [0; 1]
        // This uses a 2nd degree polynomial adjustment of the fractional part
        constexpr float POLY_D1 = 0.40196114e-7f;
        constexpr int POLY_D2 = 0xf94ee7;
        constexpr int POLY_D3 = 0x560e;

        // Compute polynomial through Horner's method
        sfpi::vFloat d1 = sfpi::vFloat(POLY_D1);
        sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(POLY_D2) + fractional_part, 0);
        sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(POLY_D3) + fractional_part, 0);
        d2 = d1 * d2;

        // Compute 2**(adjusted fractional part) through float -> int conversion
        fractional_part = _float_to_int32_exp21f_(d2 * d3);

        // Recombined exponent and mantissa: this is equivalent to 2**(x_i) * 2**(x_f)
        exponential_part = sfpi::reinterpret<sfpi::vInt>(
            sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(fractional_part), exponential_part));  // restore exponent

        y = sfpi::reinterpret<sfpi::vFloat>(exponential_part);

        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    } else {  //---exp 61f algorithm---
        v_if(val > -87.3f) {
            // The paper relies on the following formula (c.f. Section 2 and 3 of paper):
            // z = (bias + x * factor * N_m; where:
            // factor = 0x00b8aa3b (computed through log(e))
            // bias = 0x3f800000
            sfpi::vInt z = sfpu::_float_to_int32_(val * sfpi::vFloat(0x00b8aa3b) + sfpi::vFloat(0x3f800000));
            sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));         // Extract exponent
            sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract mantissa

            // Normalize mantissa field into a fractional value in [0,1)
            sfpi::vFloat frac = sfpi::int32_to_float(zif, 0) * sfpi::vFloat(1.1920929e-7f);

            // Evaluate degree-6 polynomial coefficients using Horner’s rule
            sfpi::vFloat poly = POLYVAL7(
                sfpi::vFloat(0.0002170391f),
                sfpi::vFloat(0.001243946f),
                sfpi::vFloat(0.0096788315f),
                sfpi::vFloat(0.055483369f),
                sfpi::vFloat(0.24022982f),
                sfpi::vFloat(0.69314699f),
                sfpi::vFloat(1.0000000018f),
                frac);

            // Restore exponent
            zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(poly, 127U + zii));
            y = sfpi::reinterpret<sfpi::vFloat>(zii);
        }
        v_endif;
    }

    return y;
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
            sfpi::vFloat result = _sfpu_exp_21f_<is_fp32_dest_acc_en>(val);
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

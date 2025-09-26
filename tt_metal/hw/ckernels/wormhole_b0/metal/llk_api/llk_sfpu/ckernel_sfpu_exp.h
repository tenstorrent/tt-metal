// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat sfpu_exp(sfpi::vFloat val) { return _sfpu_exp_(val); }

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
    sfpi::vFloat y = 0.0f;
    // Intermediary values can overflow if input value is below -88.0f, which leads to output increasing again instead
    // of staying at 0. This overflow happens when `log2(e) * val < 127.0f`, which correspond to `val < 88.0f`
    v_if(val > -88.0f) {
        // The paper relies on the following formula (c.f. Section 2 and 3 of paper):
        // z = (bias + x * factor * N_m; where:
        // factor = 0x00b8aa3b (computed through log(e))
        // bias = 0x3f800000
        sfpi::vInt z = sfpu::_float_to_int32_(val * sfpi::vFloat(0x00b8aa3b) + sfpi::vFloat(0x3f800000));
        sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));         // Extract exponent
        sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract mantissa

        // //---exp 21f algorithm---
        // // Polynomial coefficients for approximation of exp on [1; 2]
        // constexpr float POLY_D1 = 0.40196114e-7f;
        // constexpr int POLY_D2 = 0xf94ee7;
        // constexpr int POLY_D3 = 0x560e;

        // sfpi::vFloat d1 = sfpi::vFloat(POLY_D1);
        // sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(POLY_D2) + zif, 0);
        // sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(POLY_D3) + zif, 0);
        // d2 = d1 * d2;
        // zif = sfpu::_float_to_int32_(d2 * d3);

        // // Restore exponent
        // zii = sfpi::reinterpret<sfpi::vInt>(
        //     sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));  // restore exponent

        // y = sfpi::reinterpret<sfpi::vFloat>(zii);

        // if constexpr (!is_fp32_dest_acc_en) {
        //     // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
        //     // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
        //     // rather than 81 (which would have been correct).
        //     // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest-even.
        //     y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
        // }

        // //---exp 61f algorithm---

        // // Normalize mantissa field into a fractional value in [0,1)
        // sfpi::vFloat frac = sfpi::int32_to_float(zif, 0) * sfpi::vFloat(1.0f / float(1 << 23));

        // // Degree-6 polynomial coefficients
        // constexpr float C0 = 1.0000000018f;
        // constexpr float C1 = 0.69314699f;
        // constexpr float C2 = 0.24022982f;
        // constexpr float C3 = 0.055483369f;
        // constexpr float C4 = 0.0096788315f;
        // constexpr float C5 = 0.001243946f;
        // constexpr float C6 = 0.0002170391f;

        // // Evaluate polynomial using Horner’s rule
        // sfpi::vFloat poly =
        //     ((((((sfpi::vFloat(C6) * frac + sfpi::vFloat(C5)) * frac + sfpi::vFloat(C4)) * frac +
        //         sfpi::vFloat(C3)) *
        //             frac +
        //         sfpi::vFloat(C2)) *
        //             frac +
        //         sfpi::vFloat(C1)) *
        //             frac +
        //         sfpi::vFloat(C0));

        // // Restore exponent
        // zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(poly, 127U + zii));
        // y = sfpi::reinterpret<sfpi::vFloat>(zii);
        // if constexpr (!is_fp32_dest_acc_en) {
        //     // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
        //     // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
        //     // rather than 81 (which would have been correct).
        //     // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest-even.
        //     y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
        // }

        //---exp 41f algorithm---

        // Normalize mantissa field into a fractional value in [0,1)
        sfpi::vFloat frac = sfpi::int32_to_float(zif, 0) * sfpi::vFloat(1.0f / float(1 << 23));

        constexpr float C0 = 1.0000026f;
        constexpr float C1 = 0.69300383f;
        constexpr float C2 = 0.24144276f;
        constexpr float C3 = 0.052011463f;
        constexpr float C4 = 0.013534167f;

        sfpi::vFloat poly =
            ((((sfpi::vFloat(C4) * frac + sfpi::vFloat(C3)) * frac + sfpi::vFloat(C2)) * frac + sfpi::vFloat(C1)) *
                 frac +
             sfpi::vFloat(C0));

        // Restore exponent
        zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(poly, 127U + zii));
        y = sfpi::reinterpret<sfpi::vFloat>(zii);
        if constexpr (!is_fp32_dest_acc_en) {
            // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
            // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
            // rather than 81 (which would have been correct).
            // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest-even.
            y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
        }
    }
    v_endif;
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

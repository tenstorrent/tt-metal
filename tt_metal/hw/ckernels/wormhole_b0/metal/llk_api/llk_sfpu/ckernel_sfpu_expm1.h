// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

/*
 * This function implements expm1(x) = exp(x) - 1 using a hybrid approach to avoid
 * catastrophic cancellation when x is close to 0.
 *
 * Implementation strategy:
 * - For |x| < 0.4: Uses a 3rd-order Taylor series expansion
 *   expm1(x) ≈ x + x²/2 + x³/6, evaluated in Horner form
 * - For |x| >= 0.4: Calls _sfpu_exp_21f_ (based on the exp_21f algorithm from
 *   Moroz et al. 2022) and subtracts 1
 *
 * The Taylor series avoids the loss of precision that would occur when subtracting
 * two nearly equal numbers (exp(x) - 1) for small x.
 *
 * @param val The input value (sfpi::vFloat vector), can be any floating point number
 *
 * @return sfpi::vFloat Result of expm1(val)
 *
 * @see Moroz et al. 2022 - "Simple Multiple Precision Algorithms for Exponential Functions"
 *      ( https://doi.org/10.1109/MSP.2022.3157460 )
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_expm1_(sfpi::vFloat val) {
    sfpi::vFloat y = sfpi::vConstNeg1;
    v_if(sfpi::abs(val) < sfpi::s2vFloat16b(0.4f)) {
        // When x is very small, exp(x) is very close to 1. Hence, for improved precision, we use Taylor expansion of
        // expm1(x) = x + (x^2/2) + (x^3/6)
        // In Horner form, on reducing further : y = (val * (val * (val * 0.166f + 0.5f )+ 1)
        y = val * (sfpi::vConst1 + val * (sfpi::vFloat(0.5f) + val * sfpi::vFloat(0.166f)));
    }
    v_else {
        sfpi::vFloat exp_result = _sfpu_exp_21f_<true>(val);
        y = exp_result - sfpi::vConst1;
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
    sfpi::vFloat result;
    sfpi::vFloat abs_val = sfpi::abs(val);

    // For small |x| < 0.5: use Taylor series to avoid cancellation
    v_if(abs_val < sfpi::vConstFloatPrgm0) {
        // Use a polynomial approximation around 0 to avoid catastrophic cancellation
        // Polynomial coefficients found using Sollya with the following commands:
        // > fpminimax(exp(x)-1, [|1,2,3,4,5,6,7|], [|single...|], [-0.5; -2^(-40)] + [2^(-40); 0.5], relative);
        result = PolynomialEvaluator::eval(
            val,
            sfpi::vConst0,
            sfpi::vConst1,
            sfpi::vConstFloatPrgm1,
            0.16666667163372039794921875f,
            4.16650883853435516357421875e-2f,
            8.333188481628894805908203125e-3f,
            1.400390756316483020782470703125e-3f,
            sfpi::vConstFloatPrgm2);
    }
    v_else {
        // For moderate values: use exp(x) - 1
        // This is accurate because exp(x) is not close to 1
        // Call the accurate exp implementation and subtract 1
        sfpi::vFloat exp_result = _sfpu_exp_f32_accurate_(val);
        result = exp_result - sfpi::vConst1;
    }
    v_endif;

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

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
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
    } else {
        sfpi::vConstFloatPrgm0 = 0.5f;
        sfpi::vConstFloatPrgm1 = 0.500000059604644775390625f;
        sfpi::vConstFloatPrgm2 = 1.99588379473425447940826416015625e-4f;
    }
}

}  // namespace ckernel::sfpu

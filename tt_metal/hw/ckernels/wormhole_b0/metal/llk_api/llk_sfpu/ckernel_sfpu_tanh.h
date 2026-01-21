// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_sigmoid.h"

namespace ckernel::sfpu {

/*
 * Accurate tanh for fp32 using sigmoid: tanh(x) = 2*sigmoid(2x) - 1
 * For small |x| < 0.6, uses minimax polynomial for better accuracy
 *
 * Algorithm:
 * - For |x| < 0.6: Use minimax polynomial (Sollya-optimized)
 * - For |x| >= 0.6: Use 2*sigmoid(2x) - 1
 *
 * Target accuracy: < 5 ULP for float32 (0.5 ULP for bfloat16)
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_tanh_fp32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float POLYNOMIAL_THRESHOLD = 0.6f;

    sfpi::vFloat abs_val = sfpi::abs(val);

    v_if(abs_val < POLYNOMIAL_THRESHOLD) {
        // Small |x|: Use minimax polynomial for better accuracy
        // Polynomial coefficients found with Sollya using the following command:
        // fpminimax(tanh(x)/x, [|0,2,4,6,8|], [|single...|], [-0.6; -2^(-40)] + [2^(-40); 0.6], relative);
        sfpi::vFloat x2 = val * val;

        sfpi::vFloat p = PolynomialEvaluator::eval(
            x2,
            0.999999940395355224609375f,
            -0.33332359790802001953125f,
            0.13310669362545013427734375f,
            -5.21197654306888580322265625e-2f,
            1.5497927553951740264892578125e-2f);

        result = val * p;
    }
    v_else {
        // Normal region: Use tanh(x) = 2*sigmoid(2x) - 1
        sfpi::vFloat two_x = 2.f * val;
        sfpi::vFloat sig = _sfpu_sigmoid_<is_fp32_dest_acc_en>(two_x);

        // Compute 2*sigmoid(2x) - 1
        result = 2.f * sig - sfpi::vConst1;
    }
    v_endif;

    return result;
}

template <bool is_fp32_acc_to_dest_mode>
sfpi_inline sfpi::vFloat _sfpu_tanh_continued_fraction_(sfpi::vFloat val) {
    // Formula found at
    // https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
    // This approximation is derived from a continued fraction formula of tanh(x)

    // For negative numbers, we compute tanh(x) = -tanh(x)
    sfpi::vFloat x = sfpi::abs(val);  // set positive

    // Compute numerator and denominator of continued fraction using Horner's method
    sfpi::vFloat x2 = x * x;
    sfpi::vFloat numerator = x * (135135.f + x2 * (17326.f + x2 * (378.f + x2)));
    sfpi::vFloat denominator = PolynomialEvaluator::eval(x2, 135135.f, 62370.f, 3150.f, 28.f);

    sfpi::vFloat result = numerator * ckernel::sfpu::_sfpu_reciprocal_<2>(denominator);

    // For larger x, the continued fraction may exceed 1.0.
    // Since tanh(x) is bounded by [-1, 1], we clamp output to 1.0.
    sfpi::vFloat threshold_value = sfpi::vConst1;
    sfpi::vec_min_max(result, threshold_value);

    result = sfpi::setsgn(result, val);  // restore sign (i.e. tanh(-x) = -tanh(x))

    return result;
}

template <bool is_fp32_acc_to_dest_mode>
sfpi_inline sfpi::vFloat _sfpu_tanh_polynomial_(sfpi::vFloat x) {
    // For negative numbers, we compute tanh(-x) = -tanh(x)
    sfpi::vFloat val = sfpi::abs(x);  // set positive

    // Polynomial coefficients found using Sollya
    // val * (0.999004364013671875 + val * (3.0897438526153564453125e-2 + val * (-0.4890659749507904052734375 + val *
    // (0.281917631626129150390625 + val * (-6.6649019718170166015625e-2 + val *
    // (5.876733921468257904052734375e-3))))));
    sfpi::vFloat result = PolynomialEvaluator::eval(
        val,
        sfpi::vConst0,
        0.999004364013671875,
        3.0897438526153564453125e-2,
        -0.4890659749507904052734375,
        sfpi::vConstFloatPrgm2,
        sfpi::vConstFloatPrgm1,
        sfpi::vConstFloatPrgm0);

    // For larger x, the polynomial approximation may exceed 1.0.
    // Since tanh(x) is bounded by [-1, 1], we clamp output to 1.0.
    sfpi::vFloat threshold_value = sfpi::vConst1;
    sfpi::vec_min_max(result, threshold_value);

    result = sfpi::setsgn(result, x);  // restore sign (i.e. tanh(-x) = -tanh(x))

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanh() {
    if constexpr (APPROXIMATION_MODE) {
        // SFPU microcode
        sfpi::vUInt l0 = l_reg[sfpi::LRegs::LReg0];
        sfpi::vUInt l1 = l_reg[sfpi::LRegs::LReg1];
        sfpi::vUInt l2 = l_reg[sfpi::LRegs::LReg2];

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            val = sfpi::lut(val, l0, l1, l2);
            sfpi::dst_reg[0] = val;

            sfpi::dst_reg++;
        }

        l_reg[sfpi::LRegs::LReg0] = l0;
        l_reg[sfpi::LRegs::LReg1] = l1;
        l_reg[sfpi::LRegs::LReg2] = l2;
    } else {  // APPROXIMATION_MODE is false

        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];

            sfpi::vFloat result;

            if constexpr (is_fp32_dest_acc_en) {
                // Use accurate sigmoid-based tanh for fp32
                result = _sfpu_tanh_fp32_accurate_<is_fp32_dest_acc_en>(val);
            } else {
                result = _sfpu_tanh_polynomial_<is_fp32_dest_acc_en>(val);
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
            }

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void tanh_init() {
    if constexpr (APPROXIMATION_MODE) {
        uint imm0 = 0x1DFF;  // 0.90625*x
        uint imm1 = 0x481A;  // 0.09375*x + 0.8125
        uint imm2 = 0xFF00;  // 1
        _sfpu_load_imm16_(0, imm0);
        _sfpu_load_imm16_(1, imm1);
        _sfpu_load_imm16_(2, imm2);
    } else {
        if constexpr (is_fp32_dest_acc_en) {
            sigmoid_init<false>();
        } else {
            // Polynomial approximation
            // Store some polynomial coefficients in programmable registers
            sfpi::vConstFloatPrgm0 = 5.876733921468257904052734375e-3;
            sfpi::vConstFloatPrgm1 = -6.6649019718170166015625e-2;
            sfpi::vConstFloatPrgm2 = 0.281917631626129150390625;
        }
    }
}

}  // namespace ckernel::sfpu

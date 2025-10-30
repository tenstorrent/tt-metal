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

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_tanh_continued_fraction_(sfpi::vFloat val) {
    // Formula found at
    // https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
    // This approximation is derived from a continued fraction formula of tanh(x)

    // For negative numbers, we compute tanh(x) = -tanh(x)
    sfpi::vFloat x = sfpi::setsgn(val, 0);  // set positive

    // Compute numerator and denominator of continued fraction using Horner's method
    sfpi::vFloat x2 = x * x;
    sfpi::vFloat numerator = x * (135135.f + x2 * (17326.f + x2 * (378.f + x2)));
    sfpi::vFloat denominator = 135135.f + x2 * (62370.f + x2 * (3150.f + 28.f * x2));

    sfpi::vFloat result = ckernel::sfpu::_sfpu_reciprocal_<2>(denominator);
    result = result * numerator;

    // The limits of the continued fraction is +inf.
    // Since tanh(x) -> +inf, we clamp output to 1.0
    sfpi::vFloat threshold_value = sfpi::vConst1;
    sfpi::vec_min_max(result, threshold_value);

    result = sfpi::setsgn(result, val);  // restore sign (i.e. tanh(-x) = -tanh(x))

    if constexpr (!is_fp32_acc_to_dest_mode) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_tanh_polynomial_(sfpi::vFloat x) {
    // For negative numbers, we compute tanh(x) = -tanh(x)
    sfpi::vFloat val = sfpi::setsgn(x, 0);  // set positive

    sfpi::vFloat result = POLYVAL7<sfpi::vFloat>(
        0.999004364013671875,
        3.0897438526153564453125e-2,
        -0.4890659749507904052734375,
        sfpi::vConstFloatPrgm2,
        sfpi::vConstFloatPrgm1,
        sfpi::vConstFloatPrgm0,
        sfpi::vConst0,
        val);

    // The limits of the polynomai approximation is +inf.
    // Since tanh(x) -> +inf, we clamp output to 1.0
    sfpi::vFloat threshold_value = sfpi::vConst1;
    sfpi::vec_min_max(result, threshold_value);

    result = sfpi::setsgn(result, x);  // restore sign (i.e. tanh(-x) = -tanh(x))

    if constexpr (!is_fp32_acc_to_dest_mode) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_tanh() {
    if constexpr (APPROXIMATION_MODE) {
        // SFPU microcode
        vUInt l0 = l_reg[LRegs::LReg0];
        vUInt l1 = l_reg[LRegs::LReg1];
        vUInt l2 = l_reg[LRegs::LReg2];

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat val = dst_reg[0];
            val = lut(val, l0, l1, l2);
            dst_reg[0] = val;

            dst_reg++;
        }

        l_reg[LRegs::LReg0] = l0;
        l_reg[LRegs::LReg1] = l1;
        l_reg[LRegs::LReg2] = l2;
    } else {  // APPROXIMATION_MODE is false

        for (int d = 0; d < ITERATIONS; d++) {
            vFloat val = dst_reg[0];

            vFloat result;

            if constexpr (is_fp32_dest_acc_en) {
                result = _sfpu_tanh_continued_fraction_<is_fp32_dest_acc_en>(val);
            } else {
                result = _sfpu_tanh_polynomial_<is_fp32_dest_acc_en>(val);
            }

            dst_reg[0] = result;
            dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false>
inline void tanh_init() {
    if constexpr (APPROXIMATION_MODE) {
        uint imm0;
        uint imm1;
        uint imm2;
        imm0 = 0x1DFF;  // 0.90625*x
        imm1 = 0x481A;  // 0.09375*x + 0.8125
        imm2 = 0xFF00;  // 1
        _sfpu_load_imm16_(0, imm0);
        _sfpu_load_imm16_(1, imm1);
        _sfpu_load_imm16_(2, imm2);
    } else {
        if constexpr (is_fp32_dest_acc_en) {
            // Continue fraction
            ckernel::sfpu::_init_reciprocal_<false, false>();
        } else {
            // Polynomial approximation
            // Store some polynomial coefficients in programmable registers
            sfpi::vConstFloatPrgm0 = 5.876733921468257904052734375e-3;
            sfpi::vConstFloatPrgm1 = -6.6649019718170166015625e-2;
            sfpi::vConstFloatPrgm2 = 0.281917631626129150390625;
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_tanh() {
    // SFPU microcode
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

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

    if constexpr (!is_fp32_acc_to_dest_mode) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

template <ApproximationMode APPROX_MODE>
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
            // Continued fraction
            ckernel::sfpu::_init_sfpu_reciprocal_<false>();
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

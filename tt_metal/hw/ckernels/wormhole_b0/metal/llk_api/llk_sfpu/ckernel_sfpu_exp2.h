// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpu/ckernel_sfpu_exp2.h"

namespace ckernel::sfpu {

/**
 * This function implements binary exponentiation using a polynomial approximation algorithm
 * based on "Simple Multiple Precision Algorithms for Exponential Functions [Tips & Tricks]"
 * by Moroz et al. 2022 (https://doi.org/10.1109/MSP.2022.3157460).
 * More specifically, it is the implementation of the `exp_21f` algorithm described in Section 5
 **/

template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_exp2_21f_(sfpi::vFloat val) {
    sfpi::vFloat y = 0.0f;
    v_if(val > -127.f) {
        sfpi::vInt z = sfpu::_float_to_int32_(val * sfpi::vFloat(0x00800000) + sfpi::vFloat(0x3f800000));
        sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));         // Extract exponent
        sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract mantissa

        sfpi::vFloat d1 = sfpi::vFloat(sfpi::vConstFloatPrgm0);
        sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vConstIntPrgm1 + zif, 0);
        sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vConstIntPrgm2 + zif, 0);
        d2 = d1 * d2;
        zif = sfpu::_float_to_int32_(d2 * d3);

        zii = sfpi::reinterpret<sfpi::vInt>(
            sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));  // restore exponent

        y = sfpi::reinterpret<sfpi::vFloat>(zii);
        if constexpr (!is_fp32_dest_acc_en) {
            y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
        }
    }
    v_endif;
    return y;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_exp2() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _sfpu_exp2_21f_<is_fp32_dest_acc_en>(v);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void exp2_init() {
    sfpi::vConstFloatPrgm0 = 0.40196114e-7f;
    sfpi::vConstIntPrgm1 = 0xf94ee7;
    sfpi::vConstIntPrgm2 = 0x560e;
}

}  // namespace ckernel::sfpu

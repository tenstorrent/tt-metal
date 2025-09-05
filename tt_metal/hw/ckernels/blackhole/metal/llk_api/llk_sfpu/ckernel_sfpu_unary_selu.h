// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat _sfpu_selu_exp_21f_(sfpi::vFloat val) {
    sfpi::vFloat y = 0.0f;
    v_if(val > -88.0f) {
        sfpi::vInt z = sfpu::_float_to_int32_(val * sfpi::vFloat(0x00b8aa3b) + sfpi::vFloat(0x3f800000));
        sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));
        sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));

        sfpi::vFloat d1 = sfpi::vFloat(sfpi::vConstFloatPrgm0);
        sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vConstIntPrgm1 + zif, 0);
        sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vConstIntPrgm2 + zif, 0);
        d2 = d1 * d2;
        zif = sfpu::_float_to_int32_(d2 * d3);

        // Restore exponent
        zii = sfpi::reinterpret<sfpi::vInt>(
            sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));  // restore exponent

        y = sfpi::reinterpret<sfpi::vFloat>(zii);
    }
    v_endif;
    return y;
}

// SELU(x) = scale ∗ ( max(0, x) + min(0, α ∗ (exp(x)−1) ) )
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS>
inline void calculate_selu(uint scale, uint alpha) {
    sfpi::vFloat scale_value = Converter::as_float(scale);
    sfpi::vFloat alpha_value = Converter::as_float(alpha);
#pragma GCC unroll 8

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if(v >= 0.0f) { sfpi::dst_reg[0] = v * scale_value; }
        v_else {
            sfpi::vFloat exp_calc = _sfpu_selu_exp_21f_(v);
            sfpi::vFloat minus_mul = exp_calc - 1.0f;
            sfpi::vFloat result = minus_mul * alpha_value * scale_value;

            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
            }
            sfpi::dst_reg[0] = result;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

inline void selu_init() {
    sfpi::vConstFloatPrgm0 = 0.40196114e-7f;
    sfpi::vConstIntPrgm1 = 0xf94ee7;
    sfpi::vConstIntPrgm2 = 0x560e;
}

}  // namespace sfpu
}  // namespace ckernel

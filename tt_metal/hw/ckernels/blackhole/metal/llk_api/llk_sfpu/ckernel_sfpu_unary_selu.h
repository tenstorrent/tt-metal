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

        constexpr float POLY_D1 = 0.40196114e-7f;
        constexpr int POLY_D2 = 0xf94ee7;
        constexpr int POLY_D3 = 0x560e;

        sfpi::vFloat d1 = sfpi::vFloat(POLY_D1);
        sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(POLY_D2) + zif, 0);
        sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(POLY_D3) + zif, 0);
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
inline void calculate_selu(uint param0, uint param1) {
    sfpi::vFloat scale = Converter::as_float(param0);
    sfpi::vFloat alpha = Converter::as_float(param1);
#pragma GCC unroll 8

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if(v >= 0.0f) { sfpi::dst_reg[0] = v * scale; }
        v_else {
            sfpi::vFloat exp_calc = _sfpu_selu_exp_21f_(v);
            sfpi::vFloat minus_mul = exp_calc - 1.0f;
            sfpi::vFloat result = minus_mul * alpha * scale;

            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
            }
            sfpi::dst_reg[0] = result;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

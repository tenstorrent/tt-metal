// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel {
namespace sfpu {

template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_expm1_(sfpi::vFloat val) {
    sfpi::vFloat y = -1.0f;
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

        zii = sfpi::reinterpret<sfpi::vInt>(
            sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));  // restore exponent

        y = sfpi::reinterpret<sfpi::vFloat>(zii) - sfpi::vFloat(1.0f);

        // When x is very small, exp(x) is very close to 1. Hence, for improved precision, we use Taylor expansion of
        // expm1(x) = x + (x^2/2) + (x^3/3^2)
        v_if(sfpi::abs(val) < sfpi::vFloat(1e-3f)) {
            y = val + (sfpi::vFloat(0.5f) * val * val) + (val * val * val * sfpi::vFloat(1.166f));
        }
        v_endif;
        if constexpr (!is_fp32_dest_acc_en) {
            y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
        }
    }
    v_endif;
    return y;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_expm1() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _sfpu_expm1_<is_fp32_dest_acc_en>(v);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

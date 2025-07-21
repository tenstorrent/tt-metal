// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpu/ckernel_sfpu_exp2.h"

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat _sfpu_exp2_21f_(sfpi::vFloat val) {
    sfpi::vFloat y = 0.0f;
    v_if(val > -88.0) {
        sfpi::vInt z = sfpu::_float_to_int32_(val * sfpi::vFloat(0x00800000) + sfpi::vFloat(0x3f800000));
        sfpi::vInt zii = z & 0x7f800000;
        sfpi::vInt zif = z & sfpi::vInt(0x007fffff);  // extra mantissa

        sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
        sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif);
        sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560) + zif);
        d2 = d1 * d2;
        zif = sfpu::_float_to_int32_(d2 * d3);

        zii |= zif;  // restore exponent

        y = sfpi::reinterpret<sfpi::vFloat>(zii);
    }
    v_endif;
    return y;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_exp2() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        vFloat exp = _sfpu_exp2_21f_(v);
        dst_reg[0] = exp;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void exp2_init() {
    _init_exp2_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel

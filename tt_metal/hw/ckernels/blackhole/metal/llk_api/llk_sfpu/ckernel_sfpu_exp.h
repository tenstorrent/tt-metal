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

sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    sfpi::vFloat y = 0.0f;
    v_if(val > -88.0) {
        sfpi::vInt z = sfpu::_float_to_int32_(val * sfpi::vFloat(0x00b8aa3b) + sfpi::vFloat(0x3f800000));
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

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool SKIP_POSITIVE_CHECK = false>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_exponential_<APPROXIMATION_MODE, SCALE_EN, ITERATIONS, FAST_APPROX, SKIP_POSITIVE_CHECK>(
            exp_base_scale_factor);
    } else {
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = _sfpu_exp_21f_(val);
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

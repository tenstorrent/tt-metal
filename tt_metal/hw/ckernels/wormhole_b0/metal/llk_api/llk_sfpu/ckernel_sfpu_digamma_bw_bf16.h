// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat _digamma_bw_reciprocal_(const sfpi::vFloat in) {
    // A quadratic seed on the mantissa in [1, 2), two Newton steps and the exponent's complement.
    sfpi::vFloat negative_x = sfpi::copyman(-1.0f, in);
    sfpi::vFloat y = 1.4545459747314453f + 0.323232501745224f * negative_x;
    sfpi::vUInt scale_bits = ~sfpi::as<sfpi::vUInt>(in);
    y = 2.1212124824523926f + y * negative_x;
    sfpi::vFloat scale = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits), 0);
    sfpi::vFloat t = 1.0f + negative_x * y;
    scale *= 0.5f;
    y = y + y * t;
    t = 1.0f + negative_x * y;
    y = y + y * t;
    y = y * scale;
    return sfpi::copysgn(y, in);
}

template <int ITERATIONS = 32>
inline void calculate_digamma_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x6 = dst_reg[d];
        vFloat v5 = sfpi::abs(x6);
        vFloat v7 = v5;
        v_if(sfpi::abs(v5) < 4194304.0f) {
            vFloat biased = v5 + 12582912.0f;
            v7 = biased - 12582912.0f;
        }
        v_endif;
        vFloat v4 = v5 - v7;
        vFloat v3 = _digamma_bw_reciprocal_(v4);
        vFloat v9 = v4 * v4;
        vFloat v8 = 59.51146697998047f;
        v8 = v8 * v9 + 5.339162349700928f;
        v8 = v8 * v9 + 15.858933448791504f;
        v8 = v8 * v9 + 10.061755180358887f;
        v8 = v8 * v9 + 6.496425628662109f;
        v8 = v8 * v9 + 3.2898592948913574f;
        vFloat v2 = v3 * v3 + v8;
        vFloat v14 = v5 * vFloat(0.25f) + vFloat(0.25f);
        vFloat v13 = _digamma_bw_reciprocal_(v14);
        vFloat v12 = v13 * vFloat(0.25f);
        vFloat v11 = v12 * v12;
        vFloat v16 = -0.00333130219951272f;
        v16 = v16 * v12 + 0.020276019349694252f;
        v16 = v16 * v12 + -0.03929563984274864f;
        v16 = v16 * v12 + 0.000635107106063515f;
        v16 = v16 * v12 + 0.16665546596050262f;
        vFloat v15 = v12 * v16 + vFloat(0.5f);
        vFloat v10 = v11 * v15 + v12;
        vFloat v1 = v2 - v10;
        vFloat v18 = _digamma_bw_reciprocal_(v5);
        vFloat v17 = v18 * v18 + v10;
        vFloat factor = v17;
        vFloat x = x6;
        v_if(x < 0.0f) { factor = v1; }
        v_endif;
        v_if(
            setsgn(vFloat(dst_reg[32 + d]), 0) == 0.0f && setsgn(x, 0) < 1.0842021724855044e-19f &&
            setsgn(x, 0) != 0.0f) {
            factor = 0.0f;
        }
        v_endif;
        // DEST holds x as BF16 bits: exponent 7:0, mantissa 14:8, sign 15 (not the encoding's 0x7f80, 0x007f).
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff) {
            factor = 0.0f;
            v_if((raw & 0x8000) != 0) { factor = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
            v_if((raw & 0x7f00) != 0) { factor = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
        }
        v_endif;
        vFloat product = dst_reg[32 + d] * factor;
        v_if(setsgn(factor, 0) < 1.1754943508222875e-38f) { product = 0.0f; }
        v_endif;
        v_if(sfpi::is_nan(product)) { product = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        dst_reg[d] = convert<vFloat16b>(product, RoundMode::Nearest);
    }
}

}  // namespace ckernel::sfpu

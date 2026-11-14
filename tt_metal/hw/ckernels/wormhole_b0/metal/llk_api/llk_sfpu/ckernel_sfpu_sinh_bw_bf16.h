// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat _sinh_bw_reciprocal_(const sfpi::vFloat in) {
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
inline void calculate_sinh_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x9 = dst_reg[d];
        vFloat v8 = sfpi::abs(x9);
        vFloat v7 = v8 * vFloat(1.4426950216293335f) + vFloat(-1.0f);
        vFloat v6 = v7;
        v_if(v7 > vFloat(129.0f)) { v6 = vFloat(129.0f); }
        v_endif;
        vFloat v10 = v6;
        v_if(sfpi::abs(v6) < 4194304.0f) {
            vFloat biased = v6 + 12582912.0f;
            v10 = biased - 12582912.0f;
        }
        v_endif;
        vFloat v5 = v6 - v10;
        vFloat v4 = 0.009676037356257439f;
        v4 = v4 * v5 + 0.05592203512787819f;
        v4 = v4 * v5 + 0.2402210682630539f;
        v4 = v4 * v5 + 0.6931210160255432f;
        v4 = v4 * v5 + 1.0000001192092896f;
        vInt v3_e = sfpi::exexp(v4, sfpi::ExponentMode::Biased) + (sfpi::as<vInt>(v10 + 12582912.0f) - 1262485504);
        vFloat v3 = 0.0f;
        v_if(v3_e >= 255) { v3 = sfpi::copysgn(vFloat(std::numeric_limits<float>::infinity()), v4); }
        v_elseif(v3_e > 0) { v3 = sfpi::setexp(v4, v3_e); }
        v_endif;
        vFloat v2 = _sinh_bw_reciprocal_(v3);
        vFloat v1 = v2 * vFloat(0.25f) + v3;
        vFloat factor = v1;
        vFloat x = x9;
        v_if(setsgn(vFloat(dst_reg[32 + d]), 0) == 0.0f && setsgn(x, 0) < 712.0f) { factor = 0.0f; }
        v_endif;
        // DEST holds x as BF16 bits: exponent 7:0, mantissa 14:8, sign 15 (not the encoding's 0x7f80, 0x007f).
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff) {
            factor = std::numeric_limits<float>::infinity();
            v_if((raw & 0x8000) != 0) { factor = std::numeric_limits<float>::infinity(); }
            v_endif;
            v_if((raw & 0x7f00) != 0) { factor = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
        }
        v_endif;
        vFloat product = dst_reg[32 + d] * factor;
        v_if(sfpi::is_nan(product)) { product = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        dst_reg[d] = convert<vFloat16b>(product, RoundMode::Nearest);
    }
}

}  // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat template_reciprocal(const sfpi::vFloat in) {
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

// cosh_bw: grad times its derivative factor, as the Program template of activations/cosh_bw.json computes it.
// DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_cosh_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x10 = dst_reg[d];
        vFloat v9 = sfpi::abs(x10);
        vFloat v8 = v9 * vFloat(1.4426950216293335f) + vFloat(-1.0f);
        vFloat v7 = v8;
        v_if(v8 > vFloat(129.0f)) { v7 = vFloat(129.0f); }
        v_endif;
        vFloat v11 = v7;
        v_if(sfpi::abs(v7) < 4194304.0f) {
            vFloat biased = v7 + 12582912.0f;
            v11 = biased - 12582912.0f;
        }
        v_endif;
        vFloat v6 = v7 - v11;
        vFloat v5 = 0.009676037356257439f;
        v5 = v5 * v6 + 0.05592203512787819f;
        v5 = v5 * v6 + 0.2402210682630539f;
        v5 = v5 * v6 + 0.6931210160255432f;
        v5 = v5 * v6 + 1.0000001192092896f;
        vFloat x16 = dst_reg[d];
        vFloat v15 = sfpi::abs(x16);
        vFloat v14 = v15 * vFloat(1.4426950216293335f) + vFloat(-1.0f);
        vFloat v13 = v14;
        v_if(v14 > vFloat(129.0f)) { v13 = vFloat(129.0f); }
        v_endif;
        vFloat v12 = v13;
        v_if(sfpi::abs(v13) < 4194304.0f) {
            vFloat biased = v13 + 12582912.0f;
            v12 = biased - 12582912.0f;
        }
        v_endif;
        vInt v4_e = sfpi::exexp(v5, sfpi::ExponentMode::Biased) + (sfpi::as<vInt>(v12 + 12582912.0f) - 1262485504);
        vFloat v4 = 0.0f;
        v_if(v4_e >= 255) { v4 = sfpi::copysgn(vFloat(std::numeric_limits<float>::infinity()), v5); }
        v_elseif(v4_e > 0) { v4 = sfpi::setexp(v5, v4_e); }
        v_endif;
        vFloat v3 = template_reciprocal(v4);
        vFloat v2 = v3 * vFloat(0.25f);
        vFloat v1 = v2 - v4;
        vFloat x18 = dst_reg[d];
        vFloat v20 = x18 * x18;
        vFloat v19 = 0.008351953700184822f;
        v19 = v19 * v20 + 0.16666622459888458f;
        v19 = v19 * v20 + 1.0f;
        vFloat v17 = x18 * v19;
        vFloat v21 = v4 - v2;
        vFloat factor = v21;
        vFloat x = dst_reg[d];
        v_if(x <= -0.25f) { factor = v1; }
        v_elseif(x < 0.25f) { factor = v17; }
        v_endif;
        v_if(setsgn(vFloat(dst_reg[32 + d]), 0) == 0.0f && setsgn(x, 0) < 712.0f) { factor = 0.0f; }
        v_endif;
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff) {
            factor = std::numeric_limits<float>::infinity();
            v_if((raw & 0x8000) != 0) { factor = -std::numeric_limits<float>::infinity(); }
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

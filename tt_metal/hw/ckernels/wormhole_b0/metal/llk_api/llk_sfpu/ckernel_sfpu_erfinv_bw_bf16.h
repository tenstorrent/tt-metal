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

// erfinv_bw: grad times its derivative factor, as the Program template of activations/erfinv_bw.json computes it.
// DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_erfinv_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x7 = dst_reg[d];
        vFloat v6 = vFloat(1.0f) - x7 * x7;
        vInt e5_int = sfpi::exexp(v6);
        vFloat m5 = sfpi::setexp(v6, 127);
        v_if(m5 >= 1.4142135381698608f) {
            m5 = m5 * 0.5f;
            e5_int = e5_int + 1;
        }
        v_endif;
        vFloat e5 = sfpi::convert<vFloat>(sfpi::convert<vSMag>(e5_int), RoundMode::Nearest);
        vFloat v5 = m5;
        vFloat v4 = v5 - vFloat(1.0f);
        vFloat v8 = 0.255633145570755f;
        v8 = v8 * v4 + -0.3911224603652954f;
        v8 = v8 * v4 + 0.4852178394794464f;
        v8 = v8 * v4 + -0.7205411791801453f;
        v8 = v8 * v4 + 1.4426475763320923f;
        vFloat v3 = v4 * v8 + e5;
        vFloat v2 = 3.0244950721680652e-06f;
        v2 = v2 * v3 + 0.00010323606693418697f;
        v2 = v2 * v3 + 0.0016796003328636289f;
        v2 = v2 * v3 + 0.017537446692585945f;
        v2 = v2 * v3 + 0.1317351758480072f;
        v2 = v2 * v3 + 0.886218249797821f;
        vFloat x11 = dst_reg[d];
        vFloat v10 = vFloat(1.0f) - x11 * x11;
        vFloat v9 = template_reciprocal(v10);
        vFloat v1 = v2 * v9;
        vFloat factor = vFloat(std::numeric_limits<float>::quiet_NaN());
        vFloat x = dst_reg[d];
        v_if(x < -1.0f) { factor = vFloat(std::numeric_limits<float>::quiet_NaN()); }
        v_elseif(x <= -1.0f) { factor = vFloat(std::numeric_limits<float>::infinity()); }
        v_elseif(x < 1.0f) { factor = v1; }
        v_elseif(x <= 1.0f) { factor = vFloat(std::numeric_limits<float>::infinity()); }
        v_endif;
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff) { factor = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        vFloat product = dst_reg[32 + d] * factor;
        v_if(sfpi::is_nan(product)) { product = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        dst_reg[d] = convert<vFloat16b>(product, RoundMode::Nearest);
    }
}

}  // namespace ckernel::sfpu

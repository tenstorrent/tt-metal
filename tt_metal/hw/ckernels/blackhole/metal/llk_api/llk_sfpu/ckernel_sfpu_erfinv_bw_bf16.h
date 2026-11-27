// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat _erfinv_bw_reciprocal_(const sfpi::vFloat a) {
    // The SFPARECIP seed, then two Newton steps; a NaN step (a seed of 0 or inf) keeps the seed.
    sfpi::vFloat y = sfpi::approx_recip(a);
    sfpi::vFloat t = a * y - 2.0f;
    sfpi::vFloat y1 = y * -t - 0.0f;
    v_if(t < 0) {
        t = a * y1 - 2.0f;
        y = y1 * -t - 0.0f;
    }
    v_endif;
    return y;
}

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
        vFloat v9 = _erfinv_bw_reciprocal_(v6);
        vFloat v1 = v2 * v9;
        vFloat factor = vFloat(std::numeric_limits<float>::quiet_NaN());
        vFloat x = x7;
        v_if(x < -1.0f) { factor = vFloat(std::numeric_limits<float>::quiet_NaN()); }
        v_elseif(x <= -1.0f) { factor = vFloat(std::numeric_limits<float>::infinity()); }
        v_elseif(x < 1.0f) { factor = v1; }
        v_elseif(x <= 1.0f) { factor = vFloat(std::numeric_limits<float>::infinity()); }
        v_endif;
        // DEST holds x as BF16 bits: exponent 7:0, mantissa 14:8, sign 15 (not the encoding's 0x7f80, 0x007f).
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff) { factor = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        vFloat product = dst_reg[32 + d] * factor;
        dst_reg[d] = convert<vFloat16b>(product, RoundMode::Nearest);
    }
}

}  // namespace ckernel::sfpu

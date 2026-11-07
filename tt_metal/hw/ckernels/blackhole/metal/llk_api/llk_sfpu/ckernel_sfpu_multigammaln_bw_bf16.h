// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat _multigammaln_bw_reciprocal_(const sfpi::vFloat a) {
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
inline void calculate_multigammaln_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x7 = dst_reg[d];
        vFloat v6 = x7 - vFloat(1.5f);
        vFloat v8 = vFloat(1.0f) - x7;
        vFloat v5 = v8;
        v_if(x7 > 1.5f) { v5 = v6; }
        v_endif;
        vFloat v4 = 0.2188996821641922f;
        v4 = v4 * v5 + -0.959170937538147f;
        v4 = v4 * v5 + 1.6914705038070679f;
        v4 = v4 * v5 + -1.575974464416504f;
        v4 = v4 * v5 + 1.0264230966567993f;
        v4 = v4 * v5 + -0.9009277820587158f;
        v4 = v4 * v5 + 1.1099547147750854f;
        v4 = v4 * v5 + -1.2795804738998413f;
        v4 = v4 * v5 + 1.4609360694885254f;
        v4 = v4 * v5 + -1.9142413139343262f;
        v4 = v4 * v5 + 3.5424983501434326f;
        v4 = v4 * v5 + 1.4982531070709229f;
        v4 = v4 * v5 + -2.540717363357544f;
        v4 = v4 * v5 + -0.49999988079071045f;
        vFloat v12 = v5 * vFloat(0.5f);
        vFloat v11 = v5 * v5 + v12;
        vFloat v10 = v5;
        v_if(v5 < 1.0f) { v10 = v11; }
        v_endif;
        vFloat v9 = _multigammaln_bw_reciprocal_(v10);
        vFloat v3 = v4 * v9;
        vInt e15_int = sfpi::exexp(v5);
        vFloat m15 = sfpi::setexp(v5, 127);
        v_if(m15 >= 1.5f) {
            m15 = m15 * 0.5f;
            e15_int = e15_int + 1;
        }
        v_endif;
        vFloat e15 = sfpi::convert<vFloat>(sfpi::convert<vSMag>(e15_int), RoundMode::Nearest);
        vFloat v15 = e15;
        vFloat v17 = m15 - vFloat(1.0f);
        vFloat v18 = -0.044713906943798065f;
        v18 = v18 * v17 + 0.10456264764070511f;
        v18 = v18 * v17 + -0.13191910088062286f;
        v18 = v18 * v17 + 0.14492492377758026f;
        v18 = v18 * v17 + -0.1664535254240036f;
        v18 = v18 * v17 + 0.19989293813705444f;
        v18 = v18 * v17 + -0.250000923871994f;
        v18 = v18 * v17 + 0.3333350121974945f;
        v18 = v18 * v17 + -0.5f;
        v18 = v18 * v17 + 1.0f;
        vFloat v16 = v17 * v18;
        vFloat v14 = v15 * vFloat(0.6931471824645996f) + v16;
        vFloat v19 = -0.00713243568316102f;
        v19 = v19 * v9 + 0.0385759174823761f;
        v19 = v19 * v9 + -0.10919502377510071f;
        v19 = v19 * v9 + 0.24566347897052765f;
        v19 = v19 * v9 + -0.582656979560852f;
        v19 = v19 * v9 + 0.9999596476554871f;
        v19 = v19 * v9 + 3.9805632923162193e-07f;
        vFloat v13 = vFloat(4.0f) * v14 + v19;
        vFloat v2 = v13;
        v_if(v5 < 1.0f) { v2 = v3; }
        v_endif;
        vFloat v27 = x7 * vFloat(2.0f);
        vFloat v28 = v27;
        v_if(sfpi::abs(v27) < 4194304.0f) {
            vFloat biased = v27 + 12582912.0f;
            v28 = biased - 12582912.0f;
        }
        v_endif;
        vFloat v26 = v27 - v28;
        vFloat v25 = v26 * v26;
        vFloat v24 = vFloat(1.0f) - v25 * vFloat(4.0f);
        vFloat v29 = 1.769039273262024f;
        v29 = v29 * v25 + 0.21331503987312317f;
        v29 = v29 * v25 + 0.7517524361610413f;
        v29 = v29 * v25 + 0.6610228419303894f;
        v29 = v29 * v25 + 0.6762189269065857f;
        v29 = v29 * v25 + 0.7101264595985413f;
        v29 = v29 * v25 + 1.0f;
        vFloat v23 = v24 * v29;
        vFloat v22 = v23 * vFloat(4.0f);
        vFloat v30 = _multigammaln_bw_reciprocal_(v26);
        vFloat v21 = v2 - v22 * v30;
        vFloat v20 = vFloat(std::numeric_limits<float>::quiet_NaN());
        v_if(v26 != 0.0f) { v20 = v21; }
        v_endif;
        vFloat v1 = v20;
        v_if(x7 > 1.5f) { v1 = v2; }
        v_endif;
        vFloat factor = v1;
        vFloat x = x7;
        v_if(x < -1.1102230246251565e-16f) { factor = v1; }
        v_elseif(x <= 5.551115123125783e-17f) { factor = vFloat(std::numeric_limits<float>::quiet_NaN()); }
        v_elseif(x < 1.0f) { factor = v1; }
        v_elseif(x <= 1.0f) { factor = vFloat(-std::numeric_limits<float>::infinity()); }
        v_elseif(x < 1.5f) { factor = v1; }
        v_elseif(x <= 1.5f) { factor = vFloat(-std::numeric_limits<float>::infinity()); }
        v_endif;
        // DEST holds x as BF16 bits: exponent 7:0, mantissa 14:8, sign 15 (not the encoding's 0x7f80, 0x007f).
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff) {
            factor = std::numeric_limits<float>::infinity();
            v_if((raw & 0x8000) != 0) { factor = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
            v_if((raw & 0x7f00) != 0) { factor = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
        }
        v_endif;
        vFloat product = dst_reg[32 + d] * factor;
        dst_reg[d] = convert<vFloat16b>(product, RoundMode::Nearest);
    }
}

}  // namespace ckernel::sfpu

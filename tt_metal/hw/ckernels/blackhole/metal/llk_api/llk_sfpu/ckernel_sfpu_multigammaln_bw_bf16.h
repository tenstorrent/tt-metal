// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat template_reciprocal(const sfpi::vFloat a) {
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

// multigammaln_bw: grad times its derivative factor, as the Program template of activations/multigammaln_bw.json
// computes it. DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
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
        vFloat x14 = dst_reg[d];
        vFloat v13 = x14 - vFloat(1.5f);
        vFloat v15 = vFloat(1.0f) - x14;
        vFloat v12 = v15;
        v_if(x14 > 1.5f) { v12 = v13; }
        v_endif;
        vFloat v16 = v12 * vFloat(0.5f);
        vFloat v11 = v12 * v12 + v16;
        vFloat v10 = v12;
        v_if(v12 < 1.0f) { v10 = v11; }
        v_endif;
        vFloat v9 = template_reciprocal(v10);
        vFloat v3 = v4 * v9;
        vFloat x22 = dst_reg[d];
        vFloat v21 = x22 - vFloat(1.5f);
        vFloat v23 = vFloat(1.0f) - x22;
        vFloat v20 = v23;
        v_if(x22 > 1.5f) { v20 = v21; }
        v_endif;
        vInt e19_int = sfpi::exexp(v20);
        vFloat m19 = sfpi::setexp(v20, 127);
        v_if(m19 >= 1.5f) {
            m19 = m19 * 0.5f;
            e19_int = e19_int + 1;
        }
        v_endif;
        vFloat e19 = sfpi::convert<vFloat>(sfpi::convert<vSMag>(e19_int), RoundMode::Nearest);
        vFloat v19 = e19;
        vFloat v25 = m19 - vFloat(1.0f);
        vFloat v26 = -0.044713906943798065f;
        v26 = v26 * v25 + 0.10456264764070511f;
        v26 = v26 * v25 + -0.13191910088062286f;
        v26 = v26 * v25 + 0.14492492377758026f;
        v26 = v26 * v25 + -0.1664535254240036f;
        v26 = v26 * v25 + 0.19989293813705444f;
        v26 = v26 * v25 + -0.250000923871994f;
        v26 = v26 * v25 + 0.3333350121974945f;
        v26 = v26 * v25 + -0.5f;
        v26 = v26 * v25 + 1.0f;
        vFloat v24 = v25 * v26;
        vFloat v18 = v19 * vFloat(0.6931471824645996f) + v24;
        vFloat v27 = -0.00713243568316102f;
        v27 = v27 * v9 + 0.0385759174823761f;
        v27 = v27 * v9 + -0.10919502377510071f;
        v27 = v27 * v9 + 0.24566347897052765f;
        v27 = v27 * v9 + -0.582656979560852f;
        v27 = v27 * v9 + 0.9999596476554871f;
        v27 = v27 * v9 + 3.9805632923162193e-07f;
        vFloat v17 = vFloat(4.0f) * v18 + v27;
        vFloat v2 = v17;
        vFloat x30 = dst_reg[d];
        vFloat v29 = x30 - vFloat(1.5f);
        vFloat v31 = vFloat(1.0f) - x30;
        vFloat v28 = v31;
        v_if(x30 > 1.5f) { v28 = v29; }
        v_endif;
        v_if(v28 < 1.0f) { v2 = v3; }
        v_endif;
        vFloat x40 = dst_reg[d];
        vFloat v39 = x40 * vFloat(2.0f);
        vFloat v41 = v39;
        v_if(sfpi::abs(v39) < 4194304.0f) {
            vFloat biased = v39 + 12582912.0f;
            v41 = biased - 12582912.0f;
        }
        v_endif;
        vFloat v38 = v39 - v41;
        vFloat v37 = v38 * v38;
        vFloat v36 = vFloat(1.0f) - v37 * vFloat(4.0f);
        vFloat v42 = 1.769039273262024f;
        v42 = v42 * v37 + 0.21331503987312317f;
        v42 = v42 * v37 + 0.7517524361610413f;
        v42 = v42 * v37 + 0.6610228419303894f;
        v42 = v42 * v37 + 0.6762189269065857f;
        v42 = v42 * v37 + 0.7101264595985413f;
        v42 = v42 * v37 + 1.0f;
        vFloat v35 = v36 * v42;
        vFloat v34 = v35 * vFloat(4.0f);
        vFloat x46 = dst_reg[d];
        vFloat v45 = x46 * vFloat(2.0f);
        vFloat v47 = v45;
        v_if(sfpi::abs(v45) < 4194304.0f) {
            vFloat biased = v45 + 12582912.0f;
            v47 = biased - 12582912.0f;
        }
        v_endif;
        vFloat v44 = v45 - v47;
        vFloat v43 = template_reciprocal(v44);
        vFloat v33 = v2 - v34 * v43;
        vFloat v32 = vFloat(std::numeric_limits<float>::quiet_NaN());
        vFloat x50 = dst_reg[d];
        vFloat v49 = x50 * vFloat(2.0f);
        vFloat v51 = v49;
        v_if(sfpi::abs(v49) < 4194304.0f) {
            vFloat biased = v49 + 12582912.0f;
            v51 = biased - 12582912.0f;
        }
        v_endif;
        vFloat v48 = v49 - v51;
        v_if(v48 != 0.0f) { v32 = v33; }
        v_endif;
        vFloat v1 = v32;
        vFloat x52 = dst_reg[d];
        v_if(x52 > 1.5f) { v1 = v2; }
        v_endif;
        vFloat factor = v1;
        vFloat x = dst_reg[d];
        v_if(x < -1.1102230246251565e-16f) { factor = v1; }
        v_elseif(x <= 5.551115123125783e-17f) { factor = vFloat(std::numeric_limits<float>::quiet_NaN()); }
        v_elseif(x < 1.0f) { factor = v1; }
        v_elseif(x <= 1.0f) { factor = vFloat(-std::numeric_limits<float>::infinity()); }
        v_elseif(x < 1.5f) { factor = v1; }
        v_elseif(x <= 1.5f) { factor = vFloat(-std::numeric_limits<float>::infinity()); }
        v_endif;
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

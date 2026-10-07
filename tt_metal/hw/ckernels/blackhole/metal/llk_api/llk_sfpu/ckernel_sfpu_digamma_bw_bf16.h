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

// digamma_bw: grad times its derivative factor, as the Program template of activations/digamma_bw.json computes it.
// DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
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
        vFloat v3 = template_reciprocal(v4);
        vFloat x12 = dst_reg[d];
        vFloat v11 = sfpi::abs(x12);
        vFloat v13 = v11;
        v_if(sfpi::abs(v11) < 4194304.0f) {
            vFloat biased = v11 + 12582912.0f;
            v13 = biased - 12582912.0f;
        }
        v_endif;
        vFloat v10 = v11 - v13;
        vFloat v9 = v10 * v10;
        vFloat v8 = 59.51146697998047f;
        v8 = v8 * v9 + 5.339162349700928f;
        v8 = v8 * v9 + 15.858933448791504f;
        v8 = v8 * v9 + 10.061755180358887f;
        v8 = v8 * v9 + 6.496425628662109f;
        v8 = v8 * v9 + 3.2898592948913574f;
        vFloat v2 = v3 * v3 + v8;
        vFloat x20 = dst_reg[d];
        vFloat v19 = sfpi::abs(x20);
        vFloat v18 = v19 * vFloat(0.25f) + vFloat(0.25f);
        vFloat v17 = template_reciprocal(v18);
        vFloat v16 = v17 * vFloat(0.25f);
        vFloat v15 = v16 * v16;
        vFloat v22 = -0.00333130219951272f;
        v22 = v22 * v16 + 0.020276019349694252f;
        v22 = v22 * v16 + -0.03929563984274864f;
        v22 = v22 * v16 + 0.000635107106063515f;
        v22 = v22 * v16 + 0.16665546596050262f;
        vFloat v21 = v16 * v22 + vFloat(0.5f);
        vFloat v14 = v15 * v21 + v16;
        vFloat v1 = v2 - v14;
        vFloat x26 = dst_reg[d];
        vFloat v25 = sfpi::abs(x26);
        vFloat v24 = template_reciprocal(v25);
        vFloat v23 = v24 * v24 + v14;
        vFloat factor = v23;
        vFloat x = dst_reg[d];
        v_if(x < 0.0f) { factor = v1; }
        v_endif;
        v_if(
            setsgn(vFloat(dst_reg[32 + d]), 0) == 0.0f && setsgn(x, 0) < 1.0842021724855044e-19f &&
            setsgn(x, 0) != 0.0f) {
            factor = 0.0f;
        }
        v_endif;
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
        dst_reg[d] = convert<vFloat16b>(product, RoundMode::Nearest);
    }
}

}  // namespace ckernel::sfpu

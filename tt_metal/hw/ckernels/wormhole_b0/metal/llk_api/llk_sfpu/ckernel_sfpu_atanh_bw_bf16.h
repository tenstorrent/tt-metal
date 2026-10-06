// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

// atanh_bw: grad times c * Q(x)^k with c = 1.0, Q = -x * x + 1.0f, k = -1
// (activations/atanh_bw.json). DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_atanh_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat r;
        {
            vFloat x = dst_reg[d];
            vFloat q = -x * x + 1.0f;
            vFloat negative_x = copyman(vFloat(-1.0f), q);
            vFloat y = 1.4545459747314453f + 0.323232501745224f * negative_x;
            vUInt scale_bits = ~as<vUInt>(q);
            y = 2.1212124824523926f + y * negative_x;
            vFloat scale = setman(as<vFloat>(scale_bits), 0);
            vFloat t = 1.0f + negative_x * y;
            scale = scale * 0.5f;
            y = y + y * t;
            t = 1.0f + negative_x * y;
            y = y + y * t;
            r = copysgn(y * scale, q);
            // From 2^126 the reciprocal is below the smallest normal: 0, as the Blackhole seed gives.
            v_if(setsgn(q, 0) >= 8.507059173023462e+37f) { r = 0.0f; }
            v_endif;
        }
        vFloat factor = r;
        vFloat grad = dst_reg[32 + d];
        vFloat product = grad * factor;
        vFloat result = convert<vFloat16b>(product, RoundMode::Nearest);
        v_if(setsgn(product, 0) == 0.0f) {
            vFloat scaled = (grad * 1.8446744073709552e+19f) * factor;
            result = 0.0f;
            v_if(setsgn(scaled, 0) >= 2.1599340154984659e-19f) {
                result = copysgn(vFloat(1.1754943508222875e-38f), scaled);
            }
            v_endif;
        }
        v_endif;
        {
            vFloat x = dst_reg[d];
            v_if(setsgn(x, 0) >= 1.8446744073709552e+19f) { result = 0.0f; }
            v_endif;
            vFloat q = -x * x + 1.0f;
            v_if(setsgn(q, 0) == 0.0f) {
                vFloat infinity = sFloat16b(std::numeric_limits<float>::infinity());
                result = copysgn(infinity, grad);
                v_if(setsgn(grad, 0) == 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
                v_endif;
            }
            v_endif;
        }
        // An infinite x takes the derivative's limit, and a NaN gives NaN; the BF16 encoding selects them.
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff) {
            result = 0.0f;
            v_if((raw & 0x7f00) != 0) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
        }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu

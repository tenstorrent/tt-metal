// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

// log2_bw: grad times c * Q(x)^k with c = 1.4426950216293335, Q = x + 0.0f, k = -1
// (activations/log2_bw.json). DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_log2_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat constant;
        vFloat r;
        {
            vFloat x = dst_reg[d];
            vFloat q = x + 0.0f;
            constant = 1.4426950216293335f;
            v_if(setsgn(q, 0) >= 4.253529586511731e+37f) {
                q = q * 0.5f;
                constant = 0.7213475108146667f;
            }
            v_endif;
            vFloat y = approx_recip(q);
            y = y * (-q * y + 2.0f);
            y = y * (-q * y + 2.0f);
            r = y;
        }
        vFloat factor = r * constant;
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
            vFloat q = x + 0.0f;
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

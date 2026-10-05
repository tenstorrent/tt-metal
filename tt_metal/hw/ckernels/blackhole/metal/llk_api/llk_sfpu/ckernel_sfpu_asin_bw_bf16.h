// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

// asin_bw: grad times c * Q(x)^k with c = 1.0, Q = -x * x + 1.0f, k = -1/2
// (activations/asin_bw.json). DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_asin_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat r;
        {
            vFloat x = dst_reg[d];
            vFloat q = -x * x + 1.0f;
            vInt i = as<vInt>(as<vUInt>(q) >> 1);
            vFloat y = as<vFloat>(vInt(0x5f1110a0) - i);
            vFloat c = -y * (q * y);
            y = y * (c * (c + 2.253304958343506f) + 2.2825186252593994f);
            vFloat one_minus_xyy = -y * (q * y) + 1.0f;
            r = one_minus_xyy * addexp(y, -1) + y;
        }
        vFloat factor = r;
        vFloat grad = dst_reg[32 + d];
        vFloat product = grad * factor;
        vFloat result = convert<vFloat16b>(product, RoundMode::Nearest);
        {
            vFloat x = dst_reg[d];
            vFloat q = -x * x + 1.0f;
            v_if(q < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
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
            result = std::numeric_limits<float>::quiet_NaN();
            v_if((raw & 0x7f00) != 0) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
        }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu

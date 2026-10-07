// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

// rsqrt_bw: grad times c * Q(x)^k with c = -0.5, Q = x + 0.0f, k = -3/2
// (activations/rsqrt_bw.json). DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_rsqrt_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat r;
        {
            vFloat x = dst_reg[d];
            vFloat q = x + 0.0f;
            vInt i = as<vInt>(as<vUInt>(q) >> 1);
            vFloat y = as<vFloat>(vInt(0x5f1110a0) - i);
            vFloat c = -y * (q * y);
            y = y * (c * (c + 2.253304958343506f) + 2.2825186252593994f);
            vFloat one_minus_xyy = -y * (q * y) + 1.0f;
            r = one_minus_xyy * addexp(y, -1) + y;
        }
        vFloat factor = r * -0.5f;
        vFloat grad = dst_reg[32 + d];
        // A NaN gradient of either sign gives torch's NaN: the pole's copysgn and Wormhole's multiply
        // would keep a negative one's sign, which the pack stores as -inf.
        v_if(sfpi::is_nan(grad)) { grad = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        vFloat product = grad * factor * (r * r);
        vFloat result = convert<vFloat16b>(product, RoundMode::Nearest);
        v_if(setsgn(product, 0) == 0.0f && setsgn(grad, 0) < 1.152921504606847e+18f) {
            vFloat scaled = (grad * 1.8446744073709552e+19f) * factor * (r * r);
            result = 0.0f;
            v_if(setsgn(scaled, 0) >= 2.168404344971009e-19f) {
                result = convert<vFloat16b>(scaled, RoundMode::Nearest) * 5.421010862427522e-20f;
            }
            v_elseif(setsgn(scaled, 0) >= 2.1599340154984659e-19f) {
                result = copysgn(vFloat(1.1754943508222875e-38f), scaled);
            }
            v_endif;
        }
        v_endif;
        v_if(sfpi::is_nan(product)) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        v_if(setsgn(factor, 0) < 1.1754943508222875e-38f || setsgn(r * r, 0) < 1.1754943508222875e-38f) {
            result = 0.0f;
        }
        v_endif;
        {
            vFloat x = dst_reg[d];
            vFloat q = x + 0.0f;
            v_if(q < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
            v_if(setsgn(q, 0) == 0.0f) {
                vFloat infinity = sFloat16b(std::numeric_limits<float>::infinity());
                result = -copysgn(infinity, grad);
                v_if(setsgn(grad, 0) == 0.0f || as<vInt>(setsgn(grad, 0)) > 0x7f800000) {
                    result = std::numeric_limits<float>::quiet_NaN();
                }
                v_endif;
            }
            v_endif;
        }
        // An infinite x takes the derivative's limit, and a NaN gives NaN; the BF16 encoding selects them.
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff) {
            result = 0.0f;
            v_if(vFloat(dst_reg[d]) < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
            v_if((raw & 0x7f00) != 0) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
        }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu

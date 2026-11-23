// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 32>
inline void calculate_log10_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat r;
        {
            vFloat x = dst_reg[d];
            vFloat q = x + 0.0f;
            vFloat y = approx_recip(q);
            y = y * (-q * y + 2.0f);
            y = y * (-q * y + 2.0f);
            r = y;
        }
        vFloat factor = r * 0.4342944920063019f;
        vFloat grad = dst_reg[32 + d];
        // A NaN gradient of either sign gives torch's positive NaN: the pole's copysgn would keep a
        // negative one's sign.
        // Quasar: sfpi::is_nan spelled out; its nearby() compare lowers to an SFPIADD into read-only LREG8.
        v_if(exexp(grad, ExponentMode::Biased) >= 255 && exman(grad) != 0) {
            grad = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
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
        v_if(setsgn(factor, 0) < 1.1754943508222875e-38f) { result = 0.0f; }
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
        // An infinite x takes the derivative's limit and a NaN gives NaN.
        // Quasar: classify x by its float value, not by Blackhole's swizzled raw 16-bit Dest bits.
        vFloat x_class = dst_reg[d];
        vInt x_exp = exexp(x_class, ExponentMode::Biased);
        v_if(x_exp >= 255) {
            result = 0.0f;
            v_if(exman(x_class) != 0) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
        }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu

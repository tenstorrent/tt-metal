// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 32>
inline void calculate_logit_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat r;
        {
            vFloat x = dst_reg[d];
            vFloat q = -x * x + x;
            vFloat y = approx_recip(q);
            y = y * (-q * y + 2.0f);
            y = y * (-q * y + 2.0f);
            r = y;
        }
        vFloat factor = r;
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
        {
            vFloat x = dst_reg[d];
            vFloat q = -x * x + x;
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
        // An infinite or NaN x gives NaN.
        // Quasar: classify x by its float value, not by Blackhole's swizzled raw 16-bit Dest bits.
        vFloat x_class = dst_reg[d];
        vInt x_exp = exexp(x_class, ExponentMode::Biased);
        v_if(x_exp >= 255) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu

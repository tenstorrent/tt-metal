// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

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
        // A NaN gradient of either sign gives torch's NaN: the pole's copysgn and Wormhole's multiply
        // would keep a negative one's sign, which the pack stores as -inf.
        v_if(sfpi::is_nan(grad)) { grad = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
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
        // An infinite or NaN x gives NaN.
        // DEST holds x as BF16 bits: exponent 7:0, mantissa 14:8, sign 15 (not the encoding's 0x7f80, 0x007f).
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu

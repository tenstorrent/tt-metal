// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

// elu_bw: grad times its piecewise derivative, selected by the interval that holds x
// (activations/elu_bw.json). DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_elu_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[d];
        vFloat grad = dst_reg[32 + d];
        v_if(sfpi::is_nan(grad)) { grad = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        vFloat e0s_t = x * 1.4426950216293335f;
        v_if(e0s_t < -191.0f) { e0s_t = -191.0f; }
        v_elseif(e0s_t > 127.0f) { e0s_t = 127.0f; }
        v_endif;
        vFloat e0s_f = e0s_t + 12582912.0f;
        vFloat e0s_n = e0s_f - 12582912.0f;
        vInt e0s_k = as<vInt>(e0s_f) - 1262485504;
        vFloat e0s_r = e0s_t - e0s_n;
        vFloat e0s = 0.009560510516166687f;
        e0s = e0s * e0s_r + 0.05591703951358795f;
        e0s = e0s * e0s_r + 0.24024981260299683f;
        e0s = e0s * e0s_r + 0.6931219696998596f;
        e0s = e0s * e0s_r + 0.9999991655349731f;
        v_if(setsgn(e0s_t, 0) < 1.4901161193847656e-08f) { e0s = 1.0f; }
        v_endif;
        vInt e0s_e = exexp(e0s, ExponentMode::Biased) + e0s_k + 64;
        v_if(e0s_e <= 0) { e0s = 0.0f; }
        v_else { e0s = setexp(e0s, e0s_e); }
        v_endif;
        vFloat e0 = e0s * 5.421010862427522e-20f;
        vFloat product0 = grad * (e0);
        v_if(e0s_t < -100.0f) {
            product0 = convert<vFloat16b>(grad * (e0s), RoundMode::Nearest) * 5.421010862427522e-20f;
        }
        v_endif;
        v_if(e0s == 0.0f) { product0 = 0.0f; }
        v_endif;
        vFloat scaled0 = convert<vFloat16b>(product0, RoundMode::Nearest);
        vFloat result = grad;
        v_if(x <= 0.0f) { result = scaled0; }
        v_endif;
        // A NaN compares by its sign; select it by its BF16 encoding: exponent all ones, mantissa nonzero.
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff && (raw & 0x7f00) != 0) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu

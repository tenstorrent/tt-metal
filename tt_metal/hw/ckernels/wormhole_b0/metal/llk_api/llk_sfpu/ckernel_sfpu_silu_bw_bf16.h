// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

// silu_bw: grad times its derivative, a polynomial in x and the logistic function
// (activations/silu_bw.json). DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_silu_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
        v_if(sfpi::is_nan(vFloat(dst_reg[32 + d]))) { dst_reg[32 + d] = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        vFloat es;
        {
            vFloat a = setsgn(vFloat(dst_reg[d]), 0);
            vFloat core_t = (-a) * 1.4426950216293335f;
            v_if(core_t < -191.0f) { core_t = -191.0f; }
            v_elseif(core_t > 127.0f) { core_t = 127.0f; }
            v_endif;
            vFloat core_f = core_t + 12582912.0f;
            vFloat core_n = core_f - 12582912.0f;
            vInt core_k = as<vInt>(core_f) - 1262485504;
            vFloat core_r = core_t - core_n;
            vFloat core = 0.0013266970636323094f;
            core = core * core_r + 0.009675459936261177f;
            core = core * core_r + 0.05550742521882057f;
            core = core * core_r + 0.24022121727466583f;
            core = core * core_r + 0.6931469440460205f;
            core = core * core_r + 1.0000001192092896f;
            v_if(setsgn(core_t, 0) < 1.4901161193847656e-08f) { core = 1.0f; }
            v_endif;
            vInt core_e = exexp(core, ExponentMode::Biased) + core_k + 64;
            v_if(core_e <= 0) { core = 0.0f; }
            v_else { core = setexp(core, core_e); }
            v_endif;
            es = core;
        }
        vFloat e = es * 5.421010862427522e-20f;
        vFloat denominator = e + 1.0f;
        vFloat P = denominator * -0.5f + 1.4571068286895752f;
        P = P * (-denominator * P + 2.0f);
        P = P * (-denominator * P + 2.0f);
        P = P * (-denominator * P + 2.0f);
        v_if(e < 2.9802322387695312e-08f) { P = 1.0f; }
        v_endif;
        vFloat N = e * P;
        vFloat x = dst_reg[d];
        vFloat S = P;
        vFloat T = N;
        v_if(x < 0.0f) {
            S = N;
            T = P;
        }
        v_endif;
        vFloat product = dst_reg[32 + d] * (S * (1.0f + (T * x)));
        v_if(setsgn(x, 0) >= 69.31472778320312f && x < 0.0f) {
            vFloat Ss = es * P;
            vFloat t = dst_reg[32 + d] * (Ss * (1.0f + (T * x)));
            v_if(sfpi::is_nan(t)) { t = std::numeric_limits<float>::quiet_NaN(); }
            v_endif;
            product = convert<vFloat16b>(t, RoundMode::Nearest) * 5.421010862427522e-20f;
            v_if(es == 0.0f) { product = 0.0f; }
            v_endif;
        }
        v_endif;
        v_if(sfpi::is_nan(product)) { product = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        vFloat result = convert<vFloat16b>(product, RoundMode::Nearest);
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        // A NaN or an infinite x selects NaN by its BF16 encoding: exponent all ones.
        v_if((raw & 0x00ff) == 0x00ff) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu

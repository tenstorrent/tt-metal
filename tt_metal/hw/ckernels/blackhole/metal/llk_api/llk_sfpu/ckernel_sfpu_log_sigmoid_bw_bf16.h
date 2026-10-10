// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

// log_sigmoid_bw: grad times its derivative, a polynomial in x and the logistic function
// (activations/log_sigmoid_bw.json). DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_log_sigmoid_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
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
            vFloat core = 0.05508868396282196f;
            core = core * core_r + 0.24260404706001282f;
            core = core * core_r + 0.6932762265205383f;
            core = core * core_r + 0.9999289512634277f;
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
        vFloat product = dst_reg[32 + d] * T;
        v_if(setsgn(x, 0) >= 69.31472778320312f && x >= 0.0f) {
            vFloat Ts = es * P;
            product = convert<vFloat16b>(dst_reg[32 + d] * Ts, RoundMode::Nearest) * 5.421010862427522e-20f;
            v_if(es == 0.0f) { product = 0.0f; }
            v_endif;
        }
        v_endif;
        vFloat result = convert<vFloat16b>(product, RoundMode::Nearest);
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        // A NaN compares by its sign; select it by its BF16 encoding: exponent all ones, mantissa nonzero.
        v_if((raw & 0x00ff) == 0x00ff && (raw & 0x7f00) != 0) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 32>
inline void calculate_softplus_bw_bf16() {
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
            vFloat core = 0.05500870198011398f;
            core = core * core_r + 0.24221089482307434f;
            core = core * core_r + 0.693282961845398f;
            core = core * core_r + 1.0f;
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
        v_if(x < 0.0f) { S = N; }
        v_endif;
        vFloat product = dst_reg[32 + d] * S;
        v_if(setsgn(x, 0) >= 69.31472778320312f && x < 0.0f) {
            vFloat Ss = es * P;
            product = convert<vFloat16b>(dst_reg[32 + d] * Ss, RoundMode::Nearest) * 5.421010862427522e-20f;
            v_if(es == 0.0f) { product = 0.0f; }
            v_endif;
        }
        v_endif;
        vFloat result = convert<vFloat16b>(product, RoundMode::Nearest);
        // DEST holds x as BF16 bits: exponent 7:0, mantissa 14:8, sign 15 (not the encoding's 0x7f80, 0x007f).
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff && (raw & 0x7f00) != 0) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu

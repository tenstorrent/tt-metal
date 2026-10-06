// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <limits>
#include "sfpi.h"

namespace ckernel::sfpu {

// exp2_bw: grad times its piecewise derivative, selected by the interval that holds x
// (activations/exp2_bw.json). DEST tile idst holds x and idst + 1 holds grad; the result replaces x.
template <int ITERATIONS = 32>
inline void calculate_exp2_bw_bf16() {
    using namespace sfpi;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[d];
        vFloat grad = dst_reg[32 + d];
        vFloat g0_c = x * 1.0f + 126.47123718261719f;
        vFloat g0_z = sfpi::clamp(g0_c, 0.5f, 255.0f);
        vInt g0_m = sfpi::exman(g0_z, sfpi::MantissaMode::ImplicitOne);
        vInt g0_s = sfpi::shft(g0_m, sfpi::exexp(g0_z), sfpi::ShiftMode::Logical);
        vFloat g0_y = sfpi::as<vFloat>(g0_s);
        vMag g0_q = sfpi::exman(g0_y);
        vFloat g0_f = sfpi::convert<vFloat>(g0_q, RoundMode::Nearest);
        vFloat g0 = 2.711470501245141e-30f;
        g0 = g0 * g0_f + 8.84995218554717e-23f;
        g0 = g0 * g0_f + 3.428833204440428e-15f;
        g0 = g0 * g0_f + 8.261728368097465e-08f;
        g0 = g0 * g0_f + 1.0f;
        g0 = sfpi::setexp(g0, sfpi::exexp(g0_y, ExponentMode::Biased));
        v_if(g0_c >= 255.0f) { g0 = std::numeric_limits<float>::infinity(); }
        v_endif;
        vFloat product0 = grad * g0;
        vFloat scaled0 = convert<vFloat16b>(product0, RoundMode::Nearest);
        vFloat result = scaled0;
        // A NaN compares by its sign; select it by its BF16 encoding: exponent all ones, mantissa nonzero.
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff && (raw & 0x7f00) != 0) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        dst_reg[d] = result;
    }
}

}  // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_exp.h"
#include "cmath_common.h"
#include "ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// Quasar uses the classic two-region I1 approximation. Its single Horner
// chain keeps SFPI register pressure below the target compiler's reload limit,
// while the nonlinear exp/sqrt tables provide the asymptotic scale.
sfpi_inline sfpi::vFloat calculate_i1_small_(sfpi::vFloat x) {
    sfpi::vFloat y = x * (1.0f / 3.75f);
    y *= y;
    sfpi::vFloat poly = PolynomialEvaluator::eval<64>(
        y, 0.5f, 0.87890594f, 0.51498869f, 0.15084934f, 0.02658733f, 0.00301532f, 0.00032411f);
    return x * poly;
}

sfpi_inline sfpi::vFloat calculate_i1_large_(sfpi::vFloat abs_x, sfpi::vFloat signed_x) {
    sfpi::vFloat y = 3.75f * sfpi::approx_recip(abs_x);
    sfpi::vFloat poly = PolynomialEvaluator::eval<64>(
        y,
        0.39894228f,
        -0.03988024f,
        -0.00362018f,
        0.00163801f,
        -0.01031555f,
        0.02282967f,
        -0.02895312f,
        0.01787654f,
        -0.00420059f);
    sfpi::vFloat scale = sfpi::approx_exp(abs_x) * sfpi::approx_recip(sfpi::approx_sqrt(abs_x));
    return sfpi::copysgn(scale * poly, signed_x);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i1() {
    constexpr float I1_MAX_INPUT = 88.5f;
    constexpr float I1_THRESHOLD = 3.75f;

#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Clamp to [-88.5, 88.5] — exp() saturates near ±88.7 in FP32.
        x = sfpi::symmetric_clamp(x, I1_MAX_INPUT);

        sfpi::vFloat abs_x = sfpi::abs(x);
        sfpi::vFloat val = calculate_i1_small_(x);
        v_if(abs_x > I1_THRESHOLD) { val = calculate_i1_large_(abs_x, x); }
        v_endif;
#ifndef INP_FLOAT32
        val = sfpi::convert<sfpi::vFloat16b>(val, sfpi::RoundMode::Nearest);
#endif
        sfpi::dst_reg[0] = val;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void i1_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu

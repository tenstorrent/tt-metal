// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_tanh.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

template <int ITERATIONS>
void calculate_tanhshrink_bf16();

// tanhshrink(x) = x - tanh(x).
// For small |x|, tanh(x) ~= x, so the subtractive form x - tanh(x) suffers catastrophic
// cancellation in bf16 (both operands round to the same value -> result 0). Instead, for
// |x| <= 1 we evaluate tanhshrink directly via the factored odd polynomial x^3 * Q(x^2)
// (Q is a degree-3 minimax fit of (x - tanh(x))/x^3 on [0,1]); this preserves the x^3
// leading behaviour and has no cancellation. For |x| > 1 the cancellation is mild, so we
// return x - tanh(x). fp32 keeps the sigmoid-based accurate tanh (deg-3 would be ~1700
// fp32 ULP).
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanhshrink() {
    if constexpr (!is_fp32_dest_acc_en) {
        calculate_tanhshrink_bf16<ITERATIONS>();
        return;
    }
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat ax = sfpi::setsgn(x, 0);  // |x|

        // --- small |x| path: tanhshrink(x) = x^3 * Q(x^2) ---
        // Q is a minimax fit of (x - tanh(x))/x^3 on [0,1].
        // fp32 needs degree 6 to reach fp32 precision (degree 3 is ~1700 fp32 ULP).
        sfpi::vFloat u = x * x;
        sfpi::vFloat Q;
        Q = PolynomialEvaluator::eval(
            u,
            3.3333331347e-01f,
            -1.3333128393e-01f,
            5.3934831172e-02f,
            -2.1660288796e-02f,
            8.2192532718e-03f,
            -2.5107525289e-03f,
            4.2079269770e-04f);
        sfpi::vFloat result = x * u * Q;  // default = small path

        // --- large |x| path: x - tanh(x) ---
        v_if(ax > sfpi::vFloat(1.0f)) {
            // Evaluate on |x| (tanhshrink is odd) so the exp argument -2|x| is always
            // negative and cannot overflow; the "unsafe" exp is then safe to use.
            // Clamp |x| to 9 (tanh(9) rounds to 1.0 in fp32) so the saturation tail and
            // +/-inf stay exact and the exp argument is bounded to [-18, -2].
            sfpi::vFloat axc = ax;
            axc = sfpi::min(axc, 9.0f);
            sfpi::vFloat e = _sfpu_exp_fp32_accurate_unsafe_(-2.f * axc);
            sfpi::vFloat sig = sfpu_reciprocal_iter<2>(1.0f + e);  // sigmoid(2|x|)
            sfpi::vFloat tanh_ax = 2.f * sig - 1.0f;               // tanh(|x|)
            result = sfpi::copysgn(ax - tanh_ax, x);
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

void init_tanhshrink_bf16();

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void tanhshrink_init() {
    if constexpr (!is_fp32_dest_acc_en) {
        init_tanhshrink_bf16();
        return;
    }
    math::reset_counters(p_setrwc::SET_ABD_F);
    // The fp32 large-|x| path only needs the reciprocal Newton constants; the accurate
    // exp it uses is pure arithmetic (no LUT / programmable constants).
    sfpu_reciprocal_init<false>();
}

}  // namespace ckernel::sfpu

#include "ckernel_sfpu_tanhshrink_bf16.h"

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_tanh.h"

namespace ckernel::sfpu {

// tanhshrink(x) = x - tanh(x).
// For small |x|, tanh(x) ~= x, so the subtractive form x - tanh(x) suffers catastrophic
// cancellation in bf16 (both operands round to the same value -> result 0). Instead, for
// |x| <= 1 we evaluate tanhshrink directly via the factored odd polynomial x^3 * Q(x^2)
// (Q is a degree-3 minimax fit of (x - tanh(x))/x^3 on [0,1]); this preserves the x^3
// leading behaviour and has no cancellation. For |x| > 1 the cancellation is mild, so we
// return x - tanh(x). bf16 needs only ~2 result ULP there, so it uses a local degree-3
// tanh polynomial (cheaper than the shared deg-6 _sfpu_tanh_polynomial_); fp32 keeps the
// sigmoid-based accurate tanh (deg-3 would be ~1700 fp32 ULP).
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanhshrink() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat ax = sfpi::setsgn(x, 0);  // |x|

        // --- small |x| path: tanhshrink(x) = x^3 * Q(x^2) ---
        // Q is a minimax fit of (x - tanh(x))/x^3 on [0,1]. bf16 needs only degree 3;
        // fp32 needs degree 6 to reach fp32 precision (degree 3 is ~1700 fp32 ULP).
        sfpi::vFloat u = x * x;
        sfpi::vFloat Q;
        if constexpr (is_fp32_dest_acc_en) {
            Q = PolynomialEvaluator::eval(
                u,
                3.3333331347e-01f,
                -1.3333128393e-01f,
                5.3934831172e-02f,
                -2.1660288796e-02f,
                8.2192532718e-03f,
                -2.5107525289e-03f,
                4.2079269770e-04f);
        } else {
            Q = PolynomialEvaluator::eval(
                u, 3.3329936862e-01f, -1.3223160803e-01f, 4.8076551408e-02f, -1.0762925260e-02f);
        }
        sfpi::vFloat result = x * u * Q;  // default = small path

        // --- large |x| path: x - tanh(x) ---
        v_if(ax > sfpi::vFloat(1.0f)) {
            if constexpr (is_fp32_dest_acc_en) {
                // Evaluate on |x| (tanhshrink is odd) so the exp argument -2|x| is always
                // negative and cannot overflow; the "unsafe" exp is then safe to use.
                // Clamp |x| to 9 (tanh(9) rounds to 1.0 in fp32) so the saturation tail and
                // +/-inf stay exact and the exp argument is bounded to [-18, -2].
                sfpi::vFloat axc = ax;
                axc = sfpi::min(axc, 9.0f);
                sfpi::vFloat e = _sfpu_exp_fp32_accurate_unsafe_(-2.f * axc);
                // Denominator 1+e is provably well-conditioned here: |x|>1 so
                // e = exp(-2|x|) in (0, e^-2], giving 1+e in [1, 1+e^-2] >= 1.
                // A single Newton refinement therefore suffices (2->1 iterations).
                sfpi::vFloat sig = sfpu_reciprocal_iter<1>(1.0f + e);  // sigmoid(2|x|)
                sfpi::vFloat tanh_ax = 2.f * sig - 1.0f;               // tanh(|x|)
                result = sfpi::copysgn(ax - tanh_ax, x);
            } else {
                // tanh(|x|) via a degree-3 minimax fit on [1,3.3].
                // The poly crosses 1.0 monotonically near x~3.12 and stays >1 beyond, so the
                // clamp to 1.0 holds the saturation tail exactly (including +/-inf).
                sfpi::vFloat p = PolynomialEvaluator::eval(
                    ax, 6.1829000893e-02f, 1.0561303143e+00f, -4.0859283753e-01f, 5.3348409333e-02f);
                p = sfpi::min(p, 1.0f);
                sfpi::vFloat tanhx = sfpi::copysgn(p, x);  // tanh(-x) = -tanh(x)
                result = x - tanhx;
            }
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void tanhshrink_init() {
    // The bf16 large-|x| path uses only local literal polynomials, so it needs no init.
    if constexpr (is_fp32_dest_acc_en) {
        // The fp32 large-|x| path only needs the reciprocal Newton constants; the accurate
        // exp it uses is pure arithmetic (no LUT / programmable constants).
        sfpu_reciprocal_init<false>();
    }
}

}  // namespace ckernel::sfpu

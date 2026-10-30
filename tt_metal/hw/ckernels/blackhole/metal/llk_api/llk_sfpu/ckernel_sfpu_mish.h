// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

/**
 * Mish activation function:  mish(x) = x * tanh(softplus(x)) = x * tanh(log(1+exp(x)))
 *
 * Reducing to algebraic identity:
 *
 *     tanh(softplus(x)) = u (u + 2) / (u^2 + 2u + 2),  where u = exp(x)
 *                       = 1 - 2 / (u^2 + 2u + 2)
 * In order to avoid catastrophic cancellation for sufficiently large positive x,
 * for x >= 0, we use the rewritten form while for x < 0, we use the original form.
 *     x >= 0:  mish(x) = x * (1 - 2 / (u^2 + 2u + 2))
 *     x <  0:  mish(x) = x * u(u+2) / (u^2 + 2u + 2)
 *
 * The second form causes numerator ≈ denominator for sufficiently large positive x,
 * so any relative error in reciprocal is amplified by x. This is because sfpu_reciprocal_iter<0>
 * calls sfpi::approx_recip (~7-bit mantissa, ~0.4% relative error). By using the first form
 * the relative error in reciprocal is scaled by 2 / denom instead of by x.
 * The equivalent form x - 2x * inv_denom gives recip-error scaled as |x|·(2/denom) instead of 2/denom.
 * WH does not need this rearrangement since we use Sollya quadratic (~1e-5 relative error).
 *
 * Saturation: For x >= 8.0, mish(x) is approximated as x.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_mish() {
    constexpr float SAT_HI = 8.0f;

    // The exp's constants, loaded once and kept in LRegs across the loop (sfpi (through 7.89.0) never hoists a literal
    // out of a loop by itself); 1/ln2 is vConstFloatPrgm1 from mish_init. Only the exact fp32 arm runs the
    // Juffa exp, and with x and u live it has LRegs for two of its three constants (p1 stays a per-row
    // literal); the other three arms run exp_21f.
    constexpr bool juffa_exp = !APPROXIMATION_MODE && is_fp32_dest_acc_en;
    HoistedIf<!juffa_exp> c0 = EXP_21F_C0, c1 = EXP_21F_C1, c2 = EXP_21F_C2;
    HoistedIf<juffa_exp> neg_ln2_hi = EXP_FP32_NEG_LN2_HI, p0 = EXP_FP32_P0;
    constexpr float p1 = EXP_FP32_P1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // x >= SAT_HI: mish(x) ≈ x. For x <= -88.0f (including -inf),
        // exp(x) == 0 so mish(x) -> 0.0f; the product must not become
        // (-inf) * 0.0f = NaN.
        sfpi::vFloat result = x;

        // Clamp the deep-negative tail instead of branching on it. The NaN
        // needs an *infinite* x: with x finite, x * 0.0f is -0.0f, not NaN.
        // At x = -88.0f, u = exp(-88) ~ 6.1e-39 and the result is ~-5e-37,
        // which rounds to zero in every destination format; if u underflows
        // to 0 outright the result is -0.0f, which is equally correct.
        //
        // Branchless on purpose: an extra v_if/v_elseif here measured ~244
        // cycles/tile slower on Wormhole and adds a live value in the arm
        // the note above calls out as having LRegs for only two of its three
        // exp constants — that is also what broke the bf16 dest_acc=Yes
        // approx=No builds ("too few lregs to hold live values").
        // sfpi::max is the same guard expm1_cw_clamped uses.
        const sfpi::vFloat xc = sfpi::max(x, -88.0f);

        v_if(xc < SAT_HI) {
            sfpi::vFloat u;
            if constexpr (juffa_exp) {
                u = _sfpu_exp_fp32_accurate_prgm_<false /*unsafe*/>(xc, neg_ln2_hi, p0, p1);
            } else {
                u = _sfpu_exp_21f_bf16_prgm_<is_fp32_dest_acc_en>(xc, c0, c1, c2);
            }

            // denominator = (1 + u)^2 + 1 = u^2 + 2u + 2
            sfpi::vFloat one_plus_u = u + 1.0f;
            sfpi::vFloat denom = one_plus_u * one_plus_u + 1.0f;

            sfpi::vFloat inv_denom;
            if constexpr (APPROXIMATION_MODE) {
                inv_denom = sfpu_reciprocal_iter<0>(denom);
            } else if constexpr (is_fp32_dest_acc_en) {
                inv_denom = sfpu_reciprocal_iter<2>(denom);
            } else {
                inv_denom = sfpu_reciprocal_iter<1>(denom);
            }

            v_if(xc >= 0.0f) {
                // x * (1 - 2/denom)
                result = xc * (1.0f - 2.0f * inv_denom);
            }
            v_else {
                // Stable for x < 0: output x * u(u+2) / denom is itself small
                sfpi::vFloat numer = u * (u + 2.0f);
                // xc, not x: the clamp above is what keeps this product finite
                result = xc * (numer * inv_denom);
            }
            v_endif;
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void mish_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // calculate_mish uses the inline sfpu_reciprocal_iter<N>, not _calculate_reciprocal_internal_
    // so the SFPLOADMACRO fast-path init is not needed. But, we need sfpu_reciprocal_init's
    // vConstFloatPrgm0 = 2.0f for the inline NR step. So, call sfpu_reciprocal_init directly.
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
    // Prgm1 = 1/ln2 for the exp; the exact fp32 arm's Juffa exp reads it too (same bit pattern).
    _init_exp_hoisted_prgm_consts_();
}

}  // namespace ckernel::sfpu

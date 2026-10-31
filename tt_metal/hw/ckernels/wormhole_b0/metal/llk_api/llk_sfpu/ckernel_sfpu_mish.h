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
 * Mish activation function:  mish(x) = x * tanh(softplus(x))
 *
 * Reducing to algebraic identity:
 *
 *     tanh(softplus(x)) = u (u + 2) / (u^2 + 2u + 2),  where u = exp(x)
 *
 * so that mish becomes:
 *
 *     mish(x) = x * u (u + 2) / (u^2 + 2u + 2)
 *
 * Note: BH uses rearranged form (x - 2x/denom) for x >= 0 in order to avoid
 * cancellation through its lower-precision approx_recip.
 *
 * Saturation: For x >= 8.0, mish(x) is approximated as x.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_mish() {
    constexpr float SAT_HI = 8.0f;

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
        // that is tightest on LRegs. This keeps the single v_if below.
        // sfpi::max is the same guard expm1_cw_clamped uses.
        const sfpi::vFloat xc = sfpi::max(x, -88.0f);

        v_if(xc < SAT_HI) {
            sfpi::vFloat u;
            if constexpr (APPROXIMATION_MODE) {
                u = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(xc);
            } else {
                u = _sfpu_exp_accurate_<is_fp32_dest_acc_en>(xc);
            }

            // numerator = u * (u + 2)
            sfpi::vFloat numer = u * (u + 2.0f);

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

            // xc, not x: the clamp above is what keeps this product finite.
            result = xc * (numer * inv_denom);
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
    // exp does not need an init
    recip_init<APPROXIMATION_MODE, false>();
}

}  // namespace ckernel::sfpu

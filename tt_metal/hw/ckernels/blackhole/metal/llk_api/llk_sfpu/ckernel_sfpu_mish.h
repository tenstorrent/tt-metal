// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"

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

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // x >= SAT_HI: mish(x) ≈ x
        sfpi::vFloat result = x;

        v_if(x < SAT_HI) {
            sfpi::vFloat u;
            if constexpr (APPROXIMATION_MODE) {
                u = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(x);
            } else {
                u = _sfpu_exp_accurate_<is_fp32_dest_acc_en>(x);
            }

            // denominator = (1 + u)^2 + 1 = u^2 + 2u + 2
            sfpi::vFloat one_plus_u = u + sfpi::vConst1;
            sfpi::vFloat denom = one_plus_u * one_plus_u + sfpi::vConst1;

            sfpi::vFloat inv_denom;
            if constexpr (APPROXIMATION_MODE) {
                inv_denom = sfpu_reciprocal_iter<0>(denom);
            } else if constexpr (is_fp32_dest_acc_en) {
                inv_denom = sfpu_reciprocal_iter<2>(denom);
            } else {
                inv_denom = sfpu_reciprocal_iter<1>(denom);
            }

            v_if(x >= 0.0f) {
                // x * (1 - 2/denom)
                result = x * (1.0f - 2.0f * inv_denom);
            }
            v_else {
                // Stable for x < 0: output x * u(u+2) / denom is itself small
                sfpi::vFloat numer = u * (u + 2.0f);
                result = x * (numer * inv_denom);
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

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void mish_init() {
    // exp does not need an init.
    // calculate_mish uses the inline sfpu_reciprocal_iter<N>, not _calculate_reciprocal_internal_
    // so the SFPLOADMACRO fast-path init is not needed. But, we need sfpu_reciprocal_init's
    // vConstFloatPrgm0 = 2.0f for the inline NR step. So, call sfpu_reciprocal_init directly.
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

/**
 * Mish activation function:  mish(x) = x * tanh(softplus(x)) = x * tanh(log(1+exp(x)))
 *
 * Reducing to algebraic identity:
 *
 *     tanh(softplus(x)) = u (u + 2) / (u^2 + 2u + 2),  where u = exp(x)
 *                       = 1 - 2 / (u^2 + 2u + 2)
 * We use the second form to avoid catastrophic cancellation in finite precision.
 *     x >= 0:  mish(x) = x - 2x / (u^2 + 2u + 2)
 *     x <  0:  mish(x) = x * u(u+2) / (u^2 + 2u + 2)
 *
 * The second form causes numerator ≈ denominator for sufficiently large positive x,
 * so any relative error in reciprocal is amplified by x. This is because _sfpu_reciprocal_<0>
 * calls sfpi::approx_recip (~7-bit mantissa, ~0.4% relative error). By using the first form
 * the relative error in reciprocal is scalled by 2x / denom instead of x.

 * Saturation: For x >= 8.0, mish(x) is approximated as x.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_mish() {
    constexpr float SAT_HI = 8.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Default to the saturated value
        sfpi::vFloat result = x;

        v_if(x < SAT_HI) {
            sfpi::vFloat u;
            if constexpr (APPROXIMATION_MODE) {
                u = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(x);
            } else {
                u = _sfpu_exp_accurate_<is_fp32_dest_acc_en>(x);
            }

            // denominator = (1 + u)^2 + 1 = u^2 + 2u + 2
            sfpi::vFloat one_plus_u = u + 1.0f;
            sfpi::vFloat denom = one_plus_u * one_plus_u + 1.0f;

            sfpi::vFloat inv_denom;
            if constexpr (APPROXIMATION_MODE) {
                inv_denom = _sfpu_reciprocal_<0>(denom);
            } else if constexpr (is_fp32_dest_acc_en) {
                inv_denom = _sfpu_reciprocal_<2>(denom);
            } else {
                inv_denom = _sfpu_reciprocal_<1>(denom);
            }

            v_if(x >= 0.0f) {
                // Stable for x >= 0: correction term 2x/denom is small.
                result = x - 2.0f * x * inv_denom;
            }
            v_else {
                // Stable for x < 0: output x * u(u+2) / denom is itself small;
                sfpi::vFloat numer = u * (u + 2.0f);
                result = x * (numer * inv_denom);
            }
            v_endif;
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::NearestEven);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void mish_init() {
    // exp does not need an init.
    // Call _init_sfpu_reciprocal_ directly: calculate_mish uses _sfpu_reciprocal_<2>
    // inline (not _calculate_reciprocal_internal_), so SFPLOADMACRO fast-path init is
    // not needed. On BH, _init_reciprocal_ omits _init_sfpu_reciprocal_ (it only
    // configures SFPLOADMACRO macros), so vConstFloatPrgm0=2.0f would be unset.
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu

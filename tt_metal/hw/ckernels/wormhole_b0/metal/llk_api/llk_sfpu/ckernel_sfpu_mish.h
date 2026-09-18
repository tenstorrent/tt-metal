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
 *
 * Deep negative tail: exp(x) flushes to zero at x = ln(2^-126) = -87.3365 and the whole
 * expression then collapses to mish(x) = 0, although mish(x) -> x*exp(x) is still a normal
 * float down to x = -91.83 and is -1.02e-36 at the threshold. Below the threshold the
 * exponential is evaluated at x + 44 instead, i.e. exp(x)*e^44, and the e^-44 is applied
 * after multiplying by x, so nothing leaves the normal range. 44 is used because |x| lies in
 * [64, 128) over the whole affected band, where adding an exact integer is itself exact, so
 * the only new error is the single rounding of the e^-44 constant (5e-9 relative).
 * u(u+2)/(u^2+2u+2) equals u to far below fp32 resolution there (u < 2^-126), so the tail is
 * exactly x*exp(x).
 */
constexpr float MISH_TAIL_THRESHOLD = -87.3365f;  // ln(2^-126), where exp(x) underflows
constexpr float MISH_TAIL_SHIFT = 44.0f;
constexpr float MISH_TAIL_SCALE = 7.78113228e-20f;  // e^-44

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_mish() {
    constexpr float SAT_HI = 8.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // x >= SAT_HI: mish(x) ≈ x
        sfpi::vFloat result = x;

        v_if(x < SAT_HI) {
            sfpi::vFloat arg = x;
            v_if(x < MISH_TAIL_THRESHOLD) { arg = x + MISH_TAIL_SHIFT; }
            v_endif;

            sfpi::vFloat u;
            if constexpr (APPROXIMATION_MODE) {
                u = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(arg);
            } else {
                u = _sfpu_exp_accurate_<is_fp32_dest_acc_en>(arg);
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

            result = x * (numer * inv_denom);

            // u here is exp(x) * e^44; undo the scale after multiplying by x
            v_if(x < MISH_TAIL_THRESHOLD) { result = (x * u) * MISH_TAIL_SCALE; }
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
    // exp does not need an init
    recip_init<APPROXIMATION_MODE, false, false>();
}

}  // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_sqrt_custom.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_erf.h"
#include "ckernel_sfpu_recip.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat x) {
    // Algorithm based on "A handy approximation for the error function and its inverse" by Sergei Winitzki (2008)
    // This approximation defines erfinv(x) as:
    // erfinv(x) = sqrt( - 2/(pi*a) - log(1 - x^2)/2 + sqrt( ( 2/(pi*a) + log(1 - x^2)) ^2 - 1/a log(1 - x^2)) )
    // Where a is a polynomial coefficient used in the approximation of the error function (and reused in inverse error
    // function)

    // Compute log(1 - x^2)
    sfpi::vFloat log_value = calculate_log_body<false, false, false>(1.0f - x * x, 0);

    // Paper sets a constant a = 0.147.
    // This constant is used to compute two constant expressions:
    constexpr float TwoPiA = -4.330746750799873f;   // -2 / (pi * a)
    constexpr float OneDivA = 6.802721088435375f;  // 1/a

    // tmp = -2 / (pi * a) - log(1 - x^2)/2
    sfpi::vFloat tmp = TwoPiA + -0.5f * log_value;

    // calculated_value = temp + sqrt( temp^2 - log_value / a)
    sfpi::vFloat calculated_value = tmp * tmp - log_value * OneDivA;
    sfpi::vFloat intermediate_result = sfpu_sqrt_custom<false>(calculated_value);
    calculated_value = tmp + intermediate_result;

    sfpi::vFloat result = sfpu_sqrt_custom<false>(calculated_value);

    // --- Newton-Raphson refinement (accurate path) ---
    if constexpr (!APPROXIMATION_MODE) {
        sfpi::vFloat abs_x = sfpi::abs(x);

        sfpi::vFloat erf_est = sfpi::min(result, 10.0f);
        erf_est = piecewise_rational_eval<
            ERF_NUM_DEGREE, ERF_DEN_DEGREE, ERF_NUM_SEGMENTS, ERF_LUT_SIZE,
            true, false>(ERF_LUT, erf_est);
        erf_est = sfpi::clamp(erf_est, 0.0f, 1.0f);

        constexpr float TWO_OVER_SQRT_PI = 1.1283791671f;
        sfpi::vFloat neg_x2 = -(result * result);
        sfpi::vFloat exp_neg_x2 = _sfpu_exp_fp32_accurate_(neg_x2);
        sfpi::vFloat erf_deriv = TWO_OVER_SQRT_PI * exp_neg_x2;

        sfpi::vFloat residual = erf_est - abs_x;
        sfpi::vFloat inv_deriv = sfpu_reciprocal<false>(erf_deriv);
        result = result - residual * inv_deriv;
    }

    return result;
}

template <bool APPROXIMATION_MODE>
inline void calculate_erfinv() {
    constexpr int ITERATIONS = 8;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = calculate_erfinv_body<APPROXIMATION_MODE>(in);
        in = sfpi::dst_reg[0];  // reload due to register pressure
        sfpi::dst_reg[0] = sfpi::copysgn(result, in);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erfinv_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    log_init<false, false, false>();
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel

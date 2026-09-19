// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#if !defined(TT_POLY_LLK_DISABLE)
#include "ckernel_sfpu_erfinv_bf16.h"
#endif
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_sqrt_custom.h"

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

    // result = sqrt(calculated_value)
    sfpi::vFloat result = sfpu_sqrt_custom<false>(calculated_value);

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_erfinv_tt_poly_baseline() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = calculate_erfinv_body<false>(in);
        in = sfpi::dst_reg[0];  // reload due to register pressure
        sfpi::dst_reg[0] = sfpi::copysgn(result, in);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool use_tt_poly_bf16 = false>
inline void calculate_erfinv() {
    constexpr int ITERATIONS = 8;
#if defined(TT_POLY_LLK_DISABLE)
    calculate_erfinv_tt_poly_baseline<APPROXIMATION_MODE, ITERATIONS>();
#else
    if constexpr (use_tt_poly_bf16) {
        ckernel::sfpu::ttpoly::calculate<ttpoly_generated::ErfinvBf16Config, ITERATIONS>();
    } else {
        calculate_erfinv_tt_poly_baseline<APPROXIMATION_MODE, ITERATIONS>();
    }
#endif
}

template <bool APPROXIMATION_MODE, bool use_tt_poly_bf16 = false>
void erfinv_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
#if defined(TT_POLY_LLK_DISABLE)
    log_init<false, false, false>();
#else
    if constexpr (!use_tt_poly_bf16) {
        log_init<false, false, false>();
    }
#endif
}

}  // namespace sfpu
}  // namespace ckernel

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_sqrt_custom.h"
#include "ckernel_sfpu_exp.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat x) {
    // Algorithm based on Winitzki (2008) with precision refinement for FP32.
    // 1. Initial estimate:
    // Compute log(1 - x^2)
    sfpi::vFloat log_value = calculate_log_body<false, false, false>(1.0f - x * x, 0);

    // Constant a = 0.147
    constexpr float TwoPiA = -4.330746750799873f;  // -2 / (pi * a)
    constexpr float OneDivA = 6.802721088435375f;  // 1/a

    sfpi::vFloat tmp = TwoPiA + -0.5f * log_value;
    sfpi::vFloat calculated_value = tmp * tmp - log_value * OneDivA;
    sfpi::vFloat intermediate_result = sfpu_sqrt_custom<false, 2>(calculated_value);
    calculated_value = tmp + intermediate_result;

    sfpi::vFloat result = sfpu_sqrt_custom<false, 2>(calculated_value);

    // In accurate mode (!APPROXIMATION_MODE), apply refinement for domain edges (|x| > 0.7)
    if constexpr (!APPROXIMATION_MODE) {
        constexpr float SQRT_PI_DIV_2 = 0.886226925452758f;
        sfpi::vFloat r2 = result * result;
        sfpi::vFloat exp_r2 = _sfpu_exp_21f_bf16_unsafe_<true>(r2);
        sfpi::vFloat edge_fix = (result * SQRT_PI_DIV_2 * exp_r2);
        v_if(sfpi::abs(x) > 0.7f) {
            result = result + 0.001f * edge_fix * (sfpi::abs(x) - 0.7f);
        }
        v_endif;
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
}

}  // namespace sfpu
}  // namespace ckernel

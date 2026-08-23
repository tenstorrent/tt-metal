// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_sqrt_custom.h"
#include "ckernel_sfpu_erf.h"
#include "ckernel_sfpu_exp.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat x) {
    // Winitzki's initial estimate
    sfpi::vFloat x_sq = x * x;
    sfpi::vFloat log_value = calculate_log_body<false, false, false>(1.0f - x_sq, 0);

    constexpr float TwoPiA = -4.330746750799873f;
    constexpr float OneDivA = 6.802721088435375f;

    sfpi::vFloat tmp = TwoPiA + -0.5f * log_value;
    sfpi::vFloat calculated_value = tmp * tmp - log_value * OneDivA;
    sfpi::vFloat intermediate_result = sfpu_sqrt_custom<false>(calculated_value);
    calculated_value = tmp + intermediate_result;
    sfpi::vFloat est = sfpu_sqrt_custom<false>(calculated_value);

    // FP32 Accurate path: Refine with Newton-Raphson
    if constexpr (!APPROXIMATION_MODE && is_fp32_dest_acc_en) {
        sfpi::vFloat y = sfpi::abs(x);
        sfpi::vFloat est_sq = est * est;
        // erf(x_n)
        sfpi::vFloat erf_est = calculate_erf_body<false>(est);
        // exp(x_n^2)
        sfpi::vFloat exp_est_sq = _ckernel_sfpu_exp_accurate_<false, true>(est_sq, p_sfpu::kCONST_1_FP16B);
        
        // x_{n+1} = x_n - (erf(x_n) - y) * 0.886226925 * exp(x_n^2)
        // 0.886226925f is sqrt(pi)/2
        sfpi::vFloat err = erf_est - y;
        sfpi::vFloat correction = err * 0.886226925f * exp_est_sq;
        est = est - correction;
    }

    return est;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_erfinv() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = calculate_erfinv_body<APPROXIMATION_MODE, is_fp32_dest_acc_en>(in);
        in = sfpi::dst_reg[0];  // reload due to register pressure
        sfpi::dst_reg[0] = sfpi::copysgn(result, in);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false>
void erfinv_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    log_init<false, false, false>();
    if constexpr (!APPROXIMATION_MODE && is_fp32_dest_acc_en) {
        // erf_init also initializes reciprocal, which is safe to call
        erf_init<false>();
    }
}

}  // namespace sfpu
}  // namespace ckernel

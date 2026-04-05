// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// lgamma(x) = ln(|Gamma(x)|)
// Uses Lanczos approximation with g=5 (Numerical Recipes coefficients).
// lgamma(x) = 0.5*ln(2*pi) + (x - 0.5)*ln(x + 4.5) - (x + 4.5) + ln(series)
// where series = 1 + c1/x + c2/(x+1) + c3/(x+2) + c4/(x+3)
// Valid for x > 0. Special cases: lgamma(1) = lgamma(2) = 0.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lgamma() {
    constexpr float half_ln_2pi = 0.918938531357171f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Lanczos series: 1 + c1/x + c2/(x+1) + c3/(x+2) + c4/(x+3)
        sfpi::vFloat series = sfpi::vConst1;
        series = series + 76.18009172947146f * _sfpu_reciprocal_<1>(x);
        series = series + -86.50532032941677f * _sfpu_reciprocal_<1>(x + sfpi::vConst1);
        series = series + 24.01409824083091f * _sfpu_reciprocal_<1>(x + 2.0f);
        series = series + -1.231739572450155f * _sfpu_reciprocal_<1>(x + 3.0f);

        sfpi::vFloat t = x + 4.5f;
        sfpi::vFloat log_t = _calculate_log_body_no_init_(t);
        sfpi::vFloat log_series = _calculate_log_body_no_init_(series);

        // result = (x - 0.5) * log(t) - t + 0.5*ln(2*pi) + log(series)
        sfpi::vFloat result = (x - 0.5f) * log_t - t + half_ln_2pi + log_series;

        // Special cases: lgamma(1) = 0, lgamma(2) = 0
        v_if(x == sfpi::vConst1) { result = 0.0f; }
        v_endif;
        v_if(x == 2.0f) { result = 0.0f; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void lgamma_init() {
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu

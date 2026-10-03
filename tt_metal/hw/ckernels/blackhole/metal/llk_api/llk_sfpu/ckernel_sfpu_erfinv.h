// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_sqrt_custom.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Giles (2012), "Approximating the erfinv function": piecewise rational fit.
// Polynomial-only — evaluates on SFPU without erf primitives. Central branch
// covers w < 5; two tail branches fold through wt = sqrt(w) - 3.

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat x) {
    sfpi::vFloat log_value = calculate_log_body<false, false, false>(1.0f - x * x, 0);
    sfpi::vFloat w = -log_value;

    sfpi::vFloat p;
    v_if(w < 5.0f);
    {
        sfpi::vFloat wc = w - 2.5f;
        p =          2.81022636e-08f;
        p = p * wc + 3.43273939e-07f;
        p = p * wc - 3.5233877e-06f;
        p = p * wc - 4.39150654e-06f;
        p = p * wc + 2.1858087e-04f;
        p = p * wc - 1.25372503e-03f;
        p = p * wc - 4.17768164e-03f;
        p = p * wc + 2.46640727e-01f;
        p = p * wc + 1.50140941e+00f;
    }
    v_else;
    {
        sfpi::vFloat wt = sfpu_sqrt_custom<false>(w) - 3.0f;
        p =         -2.00214257e-04f;
        p = p * wt + 1.00950558e-04f;
        p = p * wt + 1.34934322e-03f;
        p = p * wt - 3.67342844e-03f;
        p = p * wt + 5.73950773e-03f;
        p = p * wt - 7.62246130e-03f;
        p = p * wt + 9.43887047e-03f;
        p = p * wt + 1.00167406e+00f;
        p = p * wt + 2.83297682e+00f;
    }
    v_endif;

    return x * p;
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

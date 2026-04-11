// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

#define POLYVAL5(coef4, coef3, coef2, coef1, coef0, val) \
    ((((coef4 * val + coef3) * val + coef2) * val + coef1) * val + coef0)

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_erf_body(vFloat x) {
    // Assumes x >= 0. Evaluates erf(x) for non-negative x using piecewise polynomial.
    vFloat result = 1.0f;
    v_if(x < 3.0f) {
        v_if(x >= 1.0f) { result = POLYVAL5(-0.03170029f, 0.31310241f, -1.1603072f, 1.91684792f, -0.19469693f, x); }
        v_else { result = POLYVAL5(0.166342190f, -0.476685015f, 0.0275416549, 1.12544048f, 0.0000661338118f, x); }
        v_endif;
    }
    v_endif;
    return result;
}

template <bool APPROXIMATION_MODE>
inline void calculate_erf() {
    for (int d = 0; d < 8; d++) {
        vFloat x = dst_reg[0];
        // Extract sign, compute erf(|x|), then restore sign.
        // This evaluates the polynomial ONCE instead of in both v_if/v_else branches.
        vFloat abs_x = sfpi::abs(x);
        vFloat result = calculate_erf_body<APPROXIMATION_MODE>(abs_x);
        // erf(-x) = -erf(x): apply sign of original x
        v_if(x < 0.0f) { result = -result; }
        v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_erfc() {
    for (int d = 0; d < 8; d++) {
        vFloat x = dst_reg[0];
        // Compute erf(|x|) once, then derive erfc.
        // erfc(x) = 1 - erf(x). For x < 0: erfc(x) = 1 + erf(|x|).
        vFloat abs_x = sfpi::abs(x);
        vFloat erf_val = calculate_erf_body<APPROXIMATION_MODE>(abs_x);
        vFloat result = 1.0f - erf_val;
        v_if(x < 0.0f) { result = 1.0f + erf_val; }
        v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

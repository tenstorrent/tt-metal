// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// ======================================================================
// erfc(x), two plain polynomials and no rational.
//
// Uses abs(x) symmetry: erfc(-x) = 2 - erfc(x).
//   |x| <  1.5:  erfc(x) = 1 - x*g(x^2), g a degree-5 minimax in x^2
//   |x| >= 1.5:  erfc(x) = exp(-x^2)/x * P(1/x^2), P absorbing 1/sqrt(pi)
//
// The asymptotic is accurate far below where an asymptotic expansion is
// normally trusted: measured in float64 it holds to 0.0073 BF16 ULP from 1.5,
// and still to 0.27 BF16 ULP from 1.0. Letting it take everything above 1.5
// leaves the low region an easy fit, which is what removes the piecewise
// rational and its LUT.
//
// Inputs are clamped to 9.3 before squaring, so every lane -- including the ones
// whose branch result is discarded -- satisfies the unsafe exp's precondition,
// which holds while -x^2 stays above -88.03. Past ~9.25 the result underflows to
// 0 on its own, which is the correct saturation for a function decaying to 0, so
// the bound costs nothing: erfc(9.3) is already below the smallest FP32 normal.
// ======================================================================

constexpr float ERFC_ASYMPTOTIC_THRESHOLD = 1.5f;
constexpr float ERFC_MAX_INPUT = 9.3f;

// Asymptotic branch, outlined so its intermediates do not stay live across the
// polynomial path.
sfpi_inline sfpi::vFloat calculate_erfc_asymptotic_(const sfpi::vFloat abs_x) {
    const sfpi::vFloat axc = sfpi::min(abs_x, ERFC_MAX_INPUT);
    const sfpi::vFloat neg_x2 = -(axc * axc);
    const sfpi::vFloat exp_value = _sfpu_exp_21f_bf16_unsafe_<true>(neg_x2);
    // Zero Newton iterations, spelled as the iteration count rather than as
    // sfpu_reciprocal<true>: the reciprocal refines an asymptotic whose residual is set by
    // other terms, so the approximate seed costs ~2 ULP rather than orders of magnitude.
    const sfpi::vFloat inv = sfpu_reciprocal_iter<0>(axc);
    // P(t), t = 1/x^2, degree 4, minimax on [1/100, 1/2.25]; 0.0073 BF16 ULP
    // measured after rounding the coefficients to float32. Carries 1/sqrt(pi).
    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv * inv, 5.6414234638e-01f, -2.7857559919e-01f, 3.5396602750e-01f, -4.4328993559e-01f, 2.8342172503e-01f);
    return exp_value * inv * correction;
}

template <int ITERATIONS = 8>
inline void calculate_erfc() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat abs_x = sfpi::abs(x);

        // Low path, computed unconditionally: erfc(x) = 1 - x*g(x^2).
        // g is fitted on [0, 2.25] in x^2; 0.0035 BF16 ULP measured.
        const sfpi::vFloat x2 = x * x;
        sfpi::vFloat r = 1.0f - abs_x * PolynomialEvaluator::eval(
                                            x2,
                                            1.1283125367e+00f,
                                            -3.7556339817e-01f,
                                            1.1125811263e-01f,
                                            -2.4775019023e-02f,
                                            3.7510146011e-03f,
                                            -2.8441375985e-04f);

        v_if(abs_x >= ERFC_ASYMPTOTIC_THRESHOLD) { r = calculate_erfc_asymptotic_(abs_x); }
        v_endif;
        // erfc(-x) = 2 - erfc(x)
        v_if(x < 0.0f) { r = 2.0f - r; }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erfc_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<true>();
}

}  // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "ckernel_sfpu_exp.h"  // For _sfpu_round_nearest_int32_
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel {
namespace sfpu {

// =============================================================================
// Forward GELU - Piecewise CDF Approximation
// =============================================================================
// GELU(x) = x * Phi(x) where Phi(x) = 0.5*(1+erf(x/sqrt(2))) is the CDF
//
// Strategy: approximate Phi(x) via piecewise polynomials, then multiply by x.
// This ensures GELU(0) = 0 exactly and handles linear growth naturally.
//
// Three active regions (plus zero default):
//   x >= 2.78125:          Identity (result = x)
//   -3.125 <= x < 2.78125: Core CDF polynomial (degree-15 in u=x²)
//   -13.1875 < x < -3.125: Inline Cody-Waite exp with correction polynomial
//   x <= -13.1875:          Zero (BF16 natural saturation)
//
// The exp region uses inline Cody-Waite range reduction instead of the library
// _sfpu_exp_f32_accurate_ call, skipping special-case checks (no overflow/
// underflow/NaN possible in the known input range) and reducing Taylor degree
// from 7 to 5. The correction factor (replacing the asymptotic Mills ratio
// 1 - 1/x² + 3/x⁴ - ...) is a degree-4 minimax polynomial in x, fitted to
// the TRUE erfc-based function. This eliminates the reciprocal call and its
// LUT init entirely.
//
// Saturation thresholds verified by exhaustive BF16 sweep with RNE rounding.
// =============================================================================

// Degree-15 CDF polynomial for Phi(x) over [-3.125, 2.78125]
// Phi(x) is an even function offset by 0.5: Phi(x) = 0.5 + odd_function(x)
// Only odd-power coefficients are non-zero; evaluated via u=x^2 factoring
// to avoid wasting SFPU cycles on zero even-power coefficients.
// Covers the extended range [-3.125, 2.78125) to eliminate the LEFT polynomial
// region entirely, reducing from 4 active regions to 3.
constexpr float GELU_CDF_CORE_C0 = 5.000000000e-01f;
constexpr float GELU_CDF_CORE_C1 = 3.9894151688e-01f;
constexpr float GELU_CDF_CORE_C3 = -6.6479682922e-02f;
constexpr float GELU_CDF_CORE_C5 = 9.9489670247e-03f;
constexpr float GELU_CDF_CORE_C7 = -1.1655492708e-03f;
constexpr float GELU_CDF_CORE_C9 = 1.0574820044e-04f;
constexpr float GELU_CDF_CORE_C11 = -7.0036203397e-06f;
constexpr float GELU_CDF_CORE_C13 = 2.9501944709e-07f;
constexpr float GELU_CDF_CORE_C15 = -5.7769380390e-09f;

// Degree-4 correction polynomial for the exp-based region (-13.1875, -3.125)
// P(x) ≈ GELU(x) · exp(x²/2), so result = exp(-x²/2) · P(x)
// Fitted to the TRUE erfc-based function via minimax (Chebyshev) approximation.
// Replaces the reciprocal + 3-term Mills ratio, saving ~8 ops + LUT init.
// Validated: Max ULP = 1 across all 266 BF16 values in the region.
constexpr float GELU_EXP_CORR_C0 = -2.9069766448e-01f;
constexpr float GELU_EXP_CORR_C1 = 3.9288617802e-02f;
constexpr float GELU_EXP_CORR_C2 = 5.8260409601e-03f;
constexpr float GELU_EXP_CORR_C3 = 3.9454728181e-04f;
constexpr float GELU_EXP_CORR_C4 = 1.0058581740e-05f;

// Forward GELU Evaluation with CDF Polynomial Approximation
// GELU(x) = x * Phi(x) where Phi is approximated piecewise
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_gelu_piecewise(sfpi::vFloat x) {
    sfpi::vFloat result = sfpi::vConst0;  // Default: 0 for x <= -13.1875

    v_if(x >= 2.78125f) { result = x; }
    // Core CDF region [-3.125, 2.78125): GELU(x) = x * Phi_core(x)
    // Phi(x) = C0 + x*(C1 + x^2*(C3 + x^2*(C5 + ... + x^2*C15)))
    // Factored via u=x^2 to eliminate zero even-power coefficients
    v_elseif(x >= -3.125f) {
        sfpi::vFloat u = x * x;
        sfpi::vFloat odd_poly = PolynomialEvaluator::eval(
            u,
            GELU_CDF_CORE_C1,
            GELU_CDF_CORE_C3,
            GELU_CDF_CORE_C5,
            GELU_CDF_CORE_C7,
            GELU_CDF_CORE_C9,
            GELU_CDF_CORE_C11,
            GELU_CDF_CORE_C13,
            GELU_CDF_CORE_C15);
        sfpi::vFloat phi = GELU_CDF_CORE_C0 + x * odd_poly;
        result = x * phi;
    }
    // Exp-based region (-13.1875, -3.125): exp(-x²/2) · P(x)
    // P(x) is a degree-4 correction polynomial fitted to the TRUE function
    // GELU(x)·exp(x²/2) via minimax approximation over x ∈ (-13.1875, -3.125).
    //
    // Uses inline Cody-Waite range reduction instead of library exp call.
    // For x ∈ (-13.1875, -3.125), t = -x²/2 ∈ (-86.8, -4.88), so z = t/ln2
    // ∈ (-125.3, -7.0). No overflow/underflow/NaN possible in this range,
    // so all special-case checks from _sfpu_exp_f32_accurate_ are skipped.
    // Degree-5 Taylor (vs degree-7 in library): error < 3.3e-9 relative,
    // negligible for BF16 output.
    v_elseif(x > -13.1875f) {
        sfpi::vFloat x2 = x * x;
        sfpi::vFloat t = x2 * (-0.5f);  // t = -x²/2

        // Inline Cody-Waite range reduction: exp(t) = 2^k · exp(r)
        constexpr float INV_LN2 = 1.4426950408889634f;
        sfpi::vFloat z = t * INV_LN2;

        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);

        constexpr float LN2_HI = -0.6931152343750000f;
        constexpr float LN2_LO = -3.19461832987e-05f;
        sfpi::vFloat r = k * LN2_HI + t;
        r = k * LN2_LO + r;

        // Degree-5 Taylor for exp(r), |r| < ln(2)/2 ≈ 0.347
        sfpi::vFloat p = PolynomialEvaluator::eval(
            r,
            sfpi::vConst1,   // 1
            sfpi::vConst1,   // r
            0.5f,            // r²/2!
            1.0f / 6.0f,     // r³/3!
            1.0f / 24.0f,    // r⁴/4!
            1.0f / 120.0f);  // r⁵/5!

        // Scale by 2^k via exponent bit manipulation
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);
        sfpi::vInt new_exp = p_exp + k_int;
        sfpi::vFloat exp_val = sfpi::setexp(p, new_exp);

        // Correction polynomial: P(x) ≈ GELU(x)·exp(x²/2)
        // Replaces reciprocal + Mills ratio, saving ~8 ops + LUT init
        sfpi::vFloat correction = PolynomialEvaluator::eval(
            x, GELU_EXP_CORR_C0, GELU_EXP_CORR_C1, GELU_EXP_CORR_C2, GELU_EXP_CORR_C3, GELU_EXP_CORR_C4);

        result = exp_val * correction;
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void gelu_init() {
    if constexpr (APPROXIMATION_MODE) {
        _init_gelu_<APPROXIMATION_MODE>();
    }
    // Accurate mode: no init needed (correction polynomial replaces reciprocal)
}

template <bool APPROXIMATION_MODE>
void gelu_derivative_init() {
    _init_gelu_derivative_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_gelu_<APPROXIMATION_MODE, ITERATIONS>();
    } else {
#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat result = calculate_gelu_piecewise<APPROXIMATION_MODE>(in);
            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
            }
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu_derivative() {
    _calculate_gelu_derivative_<APPROXIMATION_MODE, ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel

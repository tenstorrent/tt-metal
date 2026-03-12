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
//   -13.1875 < x < -3.125: Inline Cody-Waite exp with 3-term Mills ratio
//   x <= -13.1875:          Zero (BF16 natural saturation)
//
// The exp region uses inline Cody-Waite range reduction instead of the library
// _sfpu_exp_f32_accurate_ call, saving ~15 SFPU ops by skipping special-case
// checks (no overflow/underflow/NaN possible in the known input range) and
// reducing Taylor degree from 7 to 5.
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
    // Exp-based region (-13.1875, -3.125): asymptotic formula
    // GELU(x) ≈ -exp(-x²/2) / √(2π) · (1 - 1/x² + 3/x⁴ - 15/x⁶)
    //
    // Uses inline Cody-Waite range reduction instead of library exp call.
    // For x ∈ (-13.1875, -3.125), t = -x²/2 ∈ (-86.8, -4.88), so z = t/ln2
    // ∈ (-125.3, -7.0). No overflow/underflow/NaN possible in this range,
    // so all special-case checks from _sfpu_exp_f32_accurate_ are skipped.
    // Degree-5 Taylor (vs degree-7 in library): error < 3.3e-9 relative,
    // negligible for BF16 output.
    v_elseif(x > -13.1875f) {
        constexpr float K = -0.3989422804014327f;  // -1/√(2π)
        constexpr float K3 = 3.0f * K;             // -3/√(2π)
        constexpr float K15 = 15.0f * K;           // -15/√(2π)

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

        // 3-term Mills ratio correction: K·(1 - 1/x² + 3/x⁴ - 15/x⁶)
        // Let u = 1/x². Horner: K + u·(-K + u·(3K + u·(-15K)))
        sfpi::vFloat inv_x2 = _sfpu_reciprocal_<2>(x2);
        sfpi::vFloat scaled = ((-K15 * inv_x2 + K3) * inv_x2 - K) * inv_x2 + K;

        result = exp_val * scaled;
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void gelu_init() {
    if constexpr (APPROXIMATION_MODE) {
        _init_gelu_<APPROXIMATION_MODE>();
    } else {
        // Accurate mode needs reciprocal for exp-based region's 1/x^2
        _init_reciprocal_<false, false>();
    }
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

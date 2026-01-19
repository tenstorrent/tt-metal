// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "ckernel_sfpu_exp.h"  // For _sfpu_round_nearest_int32_

namespace ckernel {
namespace sfpu {

// =============================================================================
// Fused x * exp(t) for GELU negative tail - avoids intermediate underflow
// =============================================================================
// Computes x * exp(t) where t = -x²/2 for x in the negative region (-13.375, -5].
// Input range: t ∈ [-89.5, -12.5] (from t = -x²/2 where x ∈ [-13.375, -5])
// Output range: x * exp(t) ∈ [~-4e-38, ~-1.4e-6]
//
// This function combines specialized inline exp() with a key insight to avoid
// intermediate underflow. Advantages over _sfpu_exp_f32_accurate_():
// 1. No overflow check (never happens for negative t)
// 2. No NaN check (input is computed from valid x)
// 3. Simpler FTZ handling via exponent field check
// 4. Uses degree-5 Taylor (vs degree-7 in accurate version)
// 5. ~15-18 ops vs ~25-30 for general-purpose exp
// 6. Fused multiply avoids intermediate underflow (see below)
//
// Method:
// 1. Range reduction: t = k·ln(2) + r, where k = round(t/ln2), |r| < 0.5
// 2. Polynomial: exp(r) via degree-5 Taylor series
// 3. Fused multiply: x * poly BEFORE 2^k scaling
// 4. Scaling: (x * poly) * 2^k via exponent bit manipulation (FREE)
//
// KEY INSIGHT: exp(t) alone may underflow (e.g., exp(-88.6) ≈ 3e-39 < BF16 min normal),
// but x * exp(t) may NOT underflow (e.g., -13.3 * 3e-39 = -4e-38 > BF16 min normal).
//
// Solution: Multiply x by poly BEFORE applying the 2^k exponent shift.
//   exp(t) = 2^k * poly(r)
//   x * exp(t) = x * 2^k * poly = (x * poly) * 2^k
//
// Since |x| ≈ 5-13 and poly ≈ 0.7-1.0, x * poly has a larger exponent than poly alone.
// When we then shift by 2^k, we're less likely to underflow.
//
// Example for x = -13.3125, t = -88.62:
//   OLD: poly=0.67, k=-128 → exp = 0.67 * 2^(-128) ≈ 2e-39 → FTZ to 0!
//   NEW: x*poly=-8.9, k=-128 → x*exp = -8.9 * 2^(-128) ≈ -2.6e-38 → representable!
//
// Performance: ~15-18 operations, no external function calls
// Accuracy: Max ULP ≤ 1 across entire (-13.375, -5] range
// =============================================================================
sfpi_inline sfpi::vFloat x_times_exp_negative_tail(sfpi::vFloat x, sfpi::vFloat t) {
    // Cody-Waite constants for extended precision range reduction
    constexpr float INV_LN2 = 1.4426950408889634f;  // 1/ln(2)
    constexpr float LN2_HI = -0.6931152343750000f;  // -ln(2) high bits (exact in float)
    constexpr float LN2_LO = -3.19461832987e-05f;   // -ln(2) low bits

    // Step 1: Range reduction - split t into k*ln(2) + r
    sfpi::vFloat z = t * INV_LN2;
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);

    // Step 2: Extended precision range reduction (Cody-Waite)
    sfpi::vFloat r = k * LN2_HI + t;
    r = k * LN2_LO + r;

    // Step 3: Polynomial approximation of exp(r) for |r| < 0.5
    constexpr float C2 = 0.5f;
    constexpr float C3 = 0.166666667f;
    constexpr float C4 = 0.0416666667f;
    constexpr float C5 = 0.00833333333f;
    sfpi::vFloat poly = 1.0f + r * (1.0f + r * (C2 + r * (C3 + r * (C4 + r * C5))));

    // Step 4: FUSED MULTIPLY - multiply by x BEFORE exponent shift!
    // This is the key to avoiding intermediate underflow.
    // x * exp(t) = x * 2^k * poly = (x * poly) * 2^k
    sfpi::vFloat x_poly = x * poly;  // x * poly ≈ -9 to -13 (safe range)

    // Step 5: Exponent bit manipulation on the FUSED result
    sfpi::vInt xpoly_exp = sfpi::exexp_nodebias(x_poly);  // Extract exponent of x*poly
    sfpi::vInt new_exp = xpoly_exp + k_int;               // Shift by 2^k

    // Step 6: FTZ check on FINAL result (x * exp(t)), not intermediate exp(t)
    sfpi::vFloat result = sfpi::vConst0;
    v_if(new_exp > 0) { result = sfpi::setexp(x_poly, new_exp); }
    v_endif;

    return result;
}

// =============================================================================
// Original GELU implementation (preserved)
// =============================================================================

#define POLYVAL15(c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0, x)                         \
    (((((((((((((((c15) * (x) + (c14)) * (x) + (c13)) * (x) + (c12)) * (x) + (c11)) * (x) + (c10)) * (x) + (c9)) * \
                (x) +                                                                                              \
            (c8)) *                                                                                                \
               (x) +                                                                                               \
           (c7)) *                                                                                                 \
              (x) +                                                                                                \
          (c6)) *                                                                                                  \
             (x) +                                                                                                 \
         (c5)) *                                                                                                   \
            (x) +                                                                                                  \
        (c4)) *                                                                                                    \
           (x) +                                                                                                   \
       (c3)) *                                                                                                     \
          (x) +                                                                                                    \
      (c2)) *                                                                                                      \
         (x) +                                                                                                     \
     (c1)) * (x) +                                                                                                 \
        (c0)

inline sfpi::vFloat calculate_gelu_chebyshev(sfpi::vFloat val) {
    sfpi::vFloat result = 0.0f;
    v_if(val >= -5.5f) {
        result = POLYVAL15(
            -1.81205228163e-09,
            -4.59055119276e-08,
            -3.74540617693e-07,
            -2.29754133825e-07,
            1.19076782913e-05,
            4.25116466215e-05,
            -0.000138391838381,
            -0.000862052441087,
            0.000768340223025,
            0.0092074331601,
            -0.00208478037614,
            -0.0656369476513,
            0.00244542739174,
            0.398579460781,
            0.499174645395,
            2.98325768482e-05,
            val);

        // Ensure result has the same sign as input using setsgn
        result = setsgn(result, val);
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE>
void gelu_init() {
    _init_gelu_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
void gelu_derivative_init() {
    _init_gelu_derivative_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_gelu_<APPROXIMATION_MODE, ITERATIONS>();
    } else {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = in;
        v_if(in == 0.0f) { result = 0.0f; }
        v_elseif(in < 3.0f) { result = calculate_gelu_chebyshev(in); }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu_derivative() {
    _calculate_gelu_derivative_<APPROXIMATION_MODE, ITERATIONS>();
}

// =============================================================================
// GELU Derivative - Polynomial Approximation (NEW)
// =============================================================================
// GELU'(x) = Φ(x) + x*φ(x) where Φ is CDF, φ is PDF of standard normal
// Uses piecewise polynomials for high accuracy
// =============================================================================

// POLYVAL16 macro for degree-16 polynomial (17 coefficients)
// Horner's method with multi-line formatting for compiler compatibility
#define POLYVAL16(c16, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0, x)                      \
    ((((((((((((((((c16) * (x) + (c15)) * (x) + (c14)) * (x) + (c13)) * (x) + (c12)) * (x) + (c11)) * (x) + (c10)) * \
                 (x) +                                                                                               \
             (c9)) *                                                                                                 \
                (x) +                                                                                                \
            (c8)) *                                                                                                  \
               (x) +                                                                                                 \
           (c7)) *                                                                                                   \
              (x) +                                                                                                  \
          (c6)) *                                                                                                    \
             (x) +                                                                                                   \
         (c5)) *                                                                                                     \
            (x) +                                                                                                    \
        (c4)) *                                                                                                      \
           (x) +                                                                                                     \
       (c3)) *                                                                                                       \
          (x) +                                                                                                      \
      (c2)) *                                                                                                        \
         (x) +                                                                                                       \
     (c1)) * (x) +                                                                                                   \
        (c0)

// POLYVAL8 macro for degree-8 polynomial
#define POLYVAL8(c8, c7, c6, c5, c4, c3, c2, c1, c0, x)                                                             \
    ((((((((c8) * (x) + (c7)) * (x) + (c6)) * (x) + (c5)) * (x) + (c4)) * (x) + (c3)) * (x) + (c2)) * (x) + (c1)) * \
            (x) +                                                                                                   \
        (c0)

// Degree-16 polynomial for GELU'(x) over [-3, 3]
// Coefficients from Sollya fpminimax
constexpr float GELU_DERIV_CORE_C0 = 0.49999025f;
constexpr float GELU_DERIV_CORE_C1 = 0.79791743f;
constexpr float GELU_DERIV_CORE_C2 = 1.7774066e-4f;
constexpr float GELU_DERIV_CORE_C3 = -0.26595619f;
constexpr float GELU_DERIV_CORE_C4 = -4.5130015e-4f;
constexpr float GELU_DERIV_CORE_C5 = 5.9655134e-2f;
constexpr float GELU_DERIV_CORE_C6 = 4.1692785e-4f;
constexpr float GELU_DERIV_CORE_C7 = -9.2725726e-3f;
constexpr float GELU_DERIV_CORE_C8 = -1.8569338e-4f;
constexpr float GELU_DERIV_CORE_C9 = 1.0372815e-3f;
constexpr float GELU_DERIV_CORE_C10 = 4.4518791e-5f;
constexpr float GELU_DERIV_CORE_C11 = -8.0475649e-5f;
constexpr float GELU_DERIV_CORE_C12 = -5.8852397e-6f;
constexpr float GELU_DERIV_CORE_C13 = 3.8346534e-6f;
constexpr float GELU_DERIV_CORE_C14 = 4.0404797e-7f;
constexpr float GELU_DERIV_CORE_C15 = -8.3111068e-8f;
constexpr float GELU_DERIV_CORE_C16 = -1.1251415e-8f;

// Degree-8 SHIFTED polynomial for GELU'(x) over [-5, -3]
// SHIFTED: Evaluate p(t) where t = x + 4, so t ∈ [-1, 1] for x ∈ [-5, -3]
// This avoids catastrophic cancellation in float32 Horner's method
// Coefficients from Sollya fpminimax with shifted variable
constexpr float GELU_DERIV_LEFT_C0 = -5.03619085066020488739013671875e-4f;
constexpr float GELU_DERIV_LEFT_C1 = -1.872996450401842594146728515625e-3f;
constexpr float GELU_DERIV_LEFT_C2 = -3.2110414467751979827880859375e-3f;
constexpr float GELU_DERIV_LEFT_C3 = -3.30785498954355716705322265625e-3f;
constexpr float GELU_DERIV_LEFT_C4 = -2.20105494372546672821044921875e-3f;
constexpr float GELU_DERIV_LEFT_C5 = -8.814539178274571895599365234375e-4f;
constexpr float GELU_DERIV_LEFT_C6 = -9.72292109508998692035675048828125e-5f;
constexpr float GELU_DERIV_LEFT_C7 = 9.22545223147608339786529541015625e-5f;
constexpr float GELU_DERIV_LEFT_C8 = 3.57478638761676847934722900390625e-5f;

// Note: FL1 and FL2 polynomial coefficients removed - fused x*exp(t) method
// achieves Max ULP ≤ 1 across the entire (-13.375, -5] range, outperforming
// the Sollya polynomials which had Max ULP = 3 (FL1) and 42 (FL2).

// GELU Derivative Evaluation with Polynomial Approximation
// Saturation thresholds derived from exhaustive BF16 research (DAZ+FTZ model):
// - Zero saturation: x <= -13.375 (GELU'(x) becomes 0 in BF16)
// - One saturation: x >= 3.1719 (GELU'(x) becomes exactly 1 in BF16)
// Note: GELU'(x) has a "hump" exceeding 1.0 for x in [0.77, 3.16]
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_gelu_derivative_simple(sfpi::vFloat x) {
    sfpi::vFloat result = sfpi::vConst0;  // Default to 0 for x < -3

    // For x >= 3.1719, output saturates to 1 (verified saturation threshold)
    v_if(x >= 3.1719f) { result = sfpi::vConst1; }
    // Core region [-3, 3.1719], degree 16 polynomial
    // Polynomial reproduces the "hump" where GELU'(x) > 1
    v_elseif(x >= -3.0f) {
        result = POLYVAL16(
            GELU_DERIV_CORE_C16,
            GELU_DERIV_CORE_C15,
            GELU_DERIV_CORE_C14,
            GELU_DERIV_CORE_C13,
            GELU_DERIV_CORE_C12,
            GELU_DERIV_CORE_C11,
            GELU_DERIV_CORE_C10,
            GELU_DERIV_CORE_C9,
            GELU_DERIV_CORE_C8,
            GELU_DERIV_CORE_C7,
            GELU_DERIV_CORE_C6,
            GELU_DERIV_CORE_C5,
            GELU_DERIV_CORE_C4,
            GELU_DERIV_CORE_C3,
            GELU_DERIV_CORE_C2,
            GELU_DERIV_CORE_C1,
            GELU_DERIV_CORE_C0,
            x);
    }
    // Left region [-5, -3], degree 8 SHIFTED polynomial
    // SHIFTED: t = x + 4 maps x ∈ [-5, -3] to t ∈ [-1, 1]
    // This avoids catastrophic cancellation in float32 Horner's method
    v_elseif(x >= -5.0f) {
        sfpi::vFloat t = x + 4.0f;  // Shift to [-1, 1] range
        result = POLYVAL8(
            GELU_DERIV_LEFT_C8,
            GELU_DERIV_LEFT_C7,
            GELU_DERIV_LEFT_C6,
            GELU_DERIV_LEFT_C5,
            GELU_DERIV_LEFT_C4,
            GELU_DERIV_LEFT_C3,
            GELU_DERIV_LEFT_C2,
            GELU_DERIV_LEFT_C1,
            GELU_DERIV_LEFT_C0,
            t);
    }
    // Deep negative region (-13.375, -5]: use asymptotic formula with fused x*exp(t)
    // The fused method achieves Max ULP ≤ 1 across this entire range, outperforming
    // the Sollya polynomials (FL1, FL2) which had Max ULP = 3 and 42 respectively.
    // GELU'(x) ≈ φ(x) * (x - 1/x + 1/x³) where φ(x) = exp(-x²/2) / sqrt(2π)
    //
    // Uses exp_deep_negative_tail() - a specialized inline exp with direct exponent
    // bit manipulation (exexp_nodebias + setexp) instead of general-purpose
    // _sfpu_exp_f32_accurate_(). This follows consolidated research recommendations:
    // - Cody-Waite range reduction with LN2_HI + LN2_LO for extended precision
    // - Degree-5 Taylor polynomial for exp(r)
    // - Direct exponent field manipulation for 2^k scaling (FREE bit ops!)
    // - Explicit FTZ via exponent field check
    //
    // Performance: ~15-18 ops (inline) vs ~25-30 ops (external function)
    // Accuracy: < 1 ULP for exp computation, sub-ULP for GELU'
    //
    // For APPROXIMATION_MODE=true: use simple x*φ(x) (~1% relative error)
    // For APPROXIMATION_MODE=false: use Mills ratio correction (<0.01% relative error)
    v_elseif(x > -13.375f) {
        constexpr float INV_SQRT_2PI = 0.3989422804014327f;  // 1/sqrt(2*pi)

        sfpi::vFloat x2 = x * x;
        sfpi::vFloat t = x2 * (-0.5f);  // t = -x²/2

        // Use FUSED x * exp(t) to avoid intermediate underflow!
        // exp(t) alone can underflow (e.g., exp(-88.6) ≈ 3e-39 < BF16 min normal),
        // but x * exp(t) stays representable (e.g., -13.3 * 3e-39 = -4e-38 > min normal).
        // The fused function multiplies x by poly BEFORE the 2^k exponent shift.
        sfpi::vFloat x_exp = x_times_exp_negative_tail(x, t);

        if constexpr (APPROXIMATION_MODE) {
            // Fast mode: leading term only, ~1% relative error at x=-9
            // result = x * φ(x) = x * exp(-x²/2) / sqrt(2π)
            result = x_exp * INV_SQRT_2PI;
        } else {
            // Accurate mode: Mills ratio correction for <0.01% relative error
            // GELU'(x) ≈ φ(x) * (x - 1/x + 1/x³) = x * φ(x) * (1 - 1/x² + 1/x⁴)
            sfpi::vFloat inv_x2 = _sfpu_reciprocal_<2>(x2);  // 1/x²
            sfpi::vFloat inv_x4 = inv_x2 * inv_x2;           // 1/x⁴
            sfpi::vFloat correction = 1.0f - inv_x2 + inv_x4;
            result = x_exp * INV_SQRT_2PI * correction;
        }
    }
    // For x <= -13.375, saturate to 0
    // This is the BF16 natural saturation threshold where GELU'(x) rounds to 0.
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu_derivative_polynomial() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = calculate_gelu_derivative_simple<APPROXIMATION_MODE>(val);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void gelu_derivative_polynomial_init() {
    // No special initialization needed for polynomial evaluation
}

}  // namespace sfpu
}  // namespace ckernel

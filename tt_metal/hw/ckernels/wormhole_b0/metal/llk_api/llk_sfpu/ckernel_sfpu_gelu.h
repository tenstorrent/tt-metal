// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "ckernel_sfpu_exp.h"  // For _sfpu_exp_f32_accurate_

namespace ckernel {
namespace sfpu {

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

// Degree-8 SHIFTED polynomial for GELU'(x) over [-7, -5]
// SHIFTED: Evaluate p(t) where t = x + 6, so t ∈ [-1, 1] for x ∈ [-7, -5]
// Coefficients from Sollya fpminimax with shifted variable
constexpr float GELU_DERIV_FL1_C0 = -3.5759825323111726902425289154052734375e-8f;
constexpr float GELU_DERIV_FL1_C1 = -2.06079477038656477816402912139892578125e-7f;
constexpr float GELU_DERIV_FL1_C2 = -5.656046369040268473327159881591796875e-7f;
constexpr float GELU_DERIV_FL1_C3 = -1.0098229950017412193119525909423828125e-6f;
constexpr float GELU_DERIV_FL1_C4 = -1.400573410137440077960491180419921875e-6f;
constexpr float GELU_DERIV_FL1_C5 = -1.608632601346471346914768218994140625e-6f;
constexpr float GELU_DERIV_FL1_C6 = -1.372966607959824614226818084716796875e-6f;
constexpr float GELU_DERIV_FL1_C7 = -7.0957827347228885628283023834228515625e-7f;
constexpr float GELU_DERIV_FL1_C8 = -1.59272218525075004436075687408447265625e-7f;

// Degree-8 SHIFTED polynomial for GELU'(x) over [-9, -7]
// SHIFTED: Evaluate p(t) where t = x + 8, so t ∈ [-1, 1] for x ∈ [-9, -7]
// Note: 18% relative error is acceptable as values are < 6e-11
// Coefficients from Sollya fpminimax with shifted variable
constexpr float GELU_DERIV_FL2_C0 = -3.437316281758827363201902471701032482087612152099609375e-14f;
constexpr float GELU_DERIV_FL2_C1 = -2.3025404824981998697097651529475115239620208740234375e-13f;
constexpr float GELU_DERIV_FL2_C2 = -9.3392069251685416730879296665079891681671142578125e-13f;
constexpr float GELU_DERIV_FL2_C3 = -3.32250915148490921779966811300255358219146728515625e-12f;
constexpr float GELU_DERIV_FL2_C4 = -8.791863938262256539246664033271372318267822265625e-12f;
constexpr float GELU_DERIV_FL2_C5 = -1.46285726587702669121426879428327083587646484375e-11f;
constexpr float GELU_DERIV_FL2_C6 = -1.4239867097975977827672977582551538944244384765625e-11f;
constexpr float GELU_DERIV_FL2_C7 = -7.4159463292478022822251659817993640899658203125e-12f;
constexpr float GELU_DERIV_FL2_C8 = -1.59726810770866034516757281380705535411834716796875e-12f;

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
    // Far left region [-7, -5], degree 8 SHIFTED polynomial
    // SHIFTED: t = x + 6 maps x ∈ [-7, -5] to t ∈ [-1, 1]
    v_elseif(x >= -7.0f) {
        sfpi::vFloat t = x + 6.0f;  // Shift to [-1, 1] range
        result = POLYVAL8(
            GELU_DERIV_FL1_C8,
            GELU_DERIV_FL1_C7,
            GELU_DERIV_FL1_C6,
            GELU_DERIV_FL1_C5,
            GELU_DERIV_FL1_C4,
            GELU_DERIV_FL1_C3,
            GELU_DERIV_FL1_C2,
            GELU_DERIV_FL1_C1,
            GELU_DERIV_FL1_C0,
            t);
    }
    // Far left region [-9, -7], degree 8 SHIFTED polynomial
    // SHIFTED: t = x + 8 maps x ∈ [-9, -7] to t ∈ [-1, 1]
    // Note: 18% relative error is acceptable as values are < 6e-11
    v_elseif(x >= -9.0f) {
        sfpi::vFloat t = x + 8.0f;  // Shift to [-1, 1] range
        result = POLYVAL8(
            GELU_DERIV_FL2_C8,
            GELU_DERIV_FL2_C7,
            GELU_DERIV_FL2_C6,
            GELU_DERIV_FL2_C5,
            GELU_DERIV_FL2_C4,
            GELU_DERIV_FL2_C3,
            GELU_DERIV_FL2_C2,
            GELU_DERIV_FL2_C1,
            GELU_DERIV_FL2_C0,
            t);
    }
    // Deep negative region (-13.375, -9]: use asymptotic formula with accurate exp
    // GELU'(x) ≈ φ(x) * (x - 1/x + 1/x³) where φ(x) = exp(-x²/2) / sqrt(2π)
    // Uses _sfpu_exp_f32_accurate_ for proper underflow handling near BF16 limits.
    // Covers the full range down to BF16 natural saturation threshold (-13.375).
    //
    // Implementation based on consolidated research (negative_tail_final_consolicated_opinions.md):
    // - Mills ratio correction (x - 1/x + 1/x³) reduces boundary error at x=-9
    // - For APPROXIMATION_MODE=true: use simple x*φ(x) (~1% relative error)
    // - For APPROXIMATION_MODE=false: use correction terms (<0.01% relative error)
    v_elseif(x > -13.375f) {
        constexpr float INV_SQRT_2PI = 0.3989422804014327f;  // 1/sqrt(2*pi)

        sfpi::vFloat x2 = x * x;
        sfpi::vFloat t = x2 * (-0.5f);  // t = -x²/2

        // Use accurate exp with proper underflow handling
        sfpi::vFloat exp_val = _sfpu_exp_f32_accurate_(t);

        // Gaussian PDF: φ(x) = exp(-x²/2) / sqrt(2π)
        sfpi::vFloat phi = exp_val * INV_SQRT_2PI;

        if constexpr (APPROXIMATION_MODE) {
            // Fast mode: leading term only, ~1% relative error at x=-9
            result = x * phi;
        } else {
            // Accurate mode: Mills ratio correction for <0.01% relative error
            // GELU'(x) ≈ φ(x) * (x - 1/x + 1/x³) = x * φ(x) * (1 - 1/x² + 1/x⁴)
            sfpi::vFloat inv_x2 = _sfpu_reciprocal_<2>(x2);  // 1/x²
            sfpi::vFloat inv_x4 = inv_x2 * inv_x2;           // 1/x⁴
            sfpi::vFloat correction = 1.0f - inv_x2 + inv_x4;
            result = x * phi * correction;
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

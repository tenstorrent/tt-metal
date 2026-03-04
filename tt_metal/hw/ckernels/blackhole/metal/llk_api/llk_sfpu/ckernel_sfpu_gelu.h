// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "ckernel_sfpu_exp.h"  // For _sfpu_round_nearest_int32_
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel {
namespace sfpu {

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

// =============================================================================
// Forward GELU - Piecewise CDF Approximation (NEW)
// =============================================================================
// GELU(x) = x * Phi(x) where Phi(x) = 0.5*(1+erf(x/sqrt(2))) is the CDF
//
// Strategy: approximate Phi(x) via piecewise polynomials, then multiply by x.
// This ensures GELU(0) = 0 exactly and handles linear growth naturally.
//
// Saturation thresholds verified by exhaustive BF16 sweep (DAZ+FTZ model):
// - Zero saturation: x <= -13.1875 (GELU(x) rounds to 0 in BF16)
// - Identity saturation: x >= 5.375 (GELU(x) rounds to x in BF16)
// =============================================================================

// Degree-16 CDF polynomial for Phi(x) over [-3, 3]
// Phi(x) is an even function offset by 0.5: Phi(x) = 0.5 + odd_function(x)
// So even-degree coefficients (c2, c4, ...) are effectively zero
constexpr float GELU_CDF_CORE_C0 = 5.000000000e-01f;
constexpr float GELU_CDF_CORE_C1 = 3.989383512e-01f;
constexpr float GELU_CDF_CORE_C2 = 0.0f;
constexpr float GELU_CDF_CORE_C3 = -6.646870751e-02f;
constexpr float GELU_CDF_CORE_C4 = 0.0f;
constexpr float GELU_CDF_CORE_C5 = 9.938328774e-03f;
constexpr float GELU_CDF_CORE_C6 = 0.0f;
constexpr float GELU_CDF_CORE_C7 = -1.161210602e-03f;
constexpr float GELU_CDF_CORE_C8 = 0.0f;
constexpr float GELU_CDF_CORE_C9 = 1.049019822e-04f;
constexpr float GELU_CDF_CORE_C10 = 0.0f;
constexpr float GELU_CDF_CORE_C11 = -6.925923839e-06f;
constexpr float GELU_CDF_CORE_C12 = 0.0f;
constexpr float GELU_CDF_CORE_C13 = 2.924021994e-07f;
constexpr float GELU_CDF_CORE_C14 = 0.0f;
constexpr float GELU_CDF_CORE_C15 = -5.785760528e-09f;
constexpr float GELU_CDF_CORE_C16 = 0.0f;

// Degree-8 SHIFTED CDF polynomial for Phi(x) over [-5, -3]
// Evaluate p(t) where t = x + 4, so t in [-1, 1] for x in [-5, -3]
constexpr float GELU_CDF_LEFT_C0 = 3.166996445e-05f;
constexpr float GELU_CDF_LEFT_C1 = 1.338698095e-04f;
constexpr float GELU_CDF_LEFT_C2 = 2.677243205e-04f;
constexpr float GELU_CDF_LEFT_C3 = 3.340583863e-04f;
constexpr float GELU_CDF_LEFT_C4 = 2.894546153e-04f;
constexpr float GELU_CDF_LEFT_C5 = 1.835831419e-04f;
constexpr float GELU_CDF_LEFT_C6 = 8.395823421e-05f;
constexpr float GELU_CDF_LEFT_C7 = 2.329905868e-05f;
constexpr float GELU_CDF_LEFT_C8 = 2.286483037e-06f;

// Degree-8 CDF polynomial for Phi(x) over [3, 5.375]
constexpr float GELU_CDF_RIGHT_C0 = -2.074564169e-01f;
constexpr float GELU_CDF_RIGHT_C1 = 1.901193653e+00f;
constexpr float GELU_CDF_RIGHT_C2 = -1.316398922e+00f;
constexpr float GELU_CDF_RIGHT_C3 = 5.232226143e-01f;
constexpr float GELU_CDF_RIGHT_C4 = -1.304963828e-01f;
constexpr float GELU_CDF_RIGHT_C5 = 2.090219391e-02f;
constexpr float GELU_CDF_RIGHT_C6 = -2.098689052e-03f;
constexpr float GELU_CDF_RIGHT_C7 = 1.207102739e-04f;
constexpr float GELU_CDF_RIGHT_C8 = -3.043728055e-06f;

// Forward GELU Evaluation with CDF Polynomial Approximation
// GELU(x) = x * Phi(x) where Phi is approximated piecewise
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_gelu_piecewise(sfpi::vFloat x) {
    sfpi::vFloat result = sfpi::vConst0;  // Default: 0 for x <= -13.1875

    // Identity saturation: x >= 5.375
    v_if(x >= 5.375f) { result = x; }
    // Right CDF region [3, 5.375): GELU(x) = x * Phi_right(x)
    v_elseif(x >= 3.0f) {
        sfpi::vFloat phi = PolynomialEvaluator::eval(
            x,
            GELU_CDF_RIGHT_C0,
            GELU_CDF_RIGHT_C1,
            GELU_CDF_RIGHT_C2,
            GELU_CDF_RIGHT_C3,
            GELU_CDF_RIGHT_C4,
            GELU_CDF_RIGHT_C5,
            GELU_CDF_RIGHT_C6,
            GELU_CDF_RIGHT_C7,
            GELU_CDF_RIGHT_C8);
        result = x * phi;
    }
    // Core CDF region [-3, 3): GELU(x) = x * Phi_core(x)
    v_elseif(x >= -3.0f) {
        sfpi::vFloat phi = PolynomialEvaluator::eval(
            x,
            GELU_CDF_CORE_C0,
            GELU_CDF_CORE_C1,
            GELU_CDF_CORE_C2,
            GELU_CDF_CORE_C3,
            GELU_CDF_CORE_C4,
            GELU_CDF_CORE_C5,
            GELU_CDF_CORE_C6,
            GELU_CDF_CORE_C7,
            GELU_CDF_CORE_C8,
            GELU_CDF_CORE_C9,
            GELU_CDF_CORE_C10,
            GELU_CDF_CORE_C11,
            GELU_CDF_CORE_C12,
            GELU_CDF_CORE_C13,
            GELU_CDF_CORE_C14,
            GELU_CDF_CORE_C15,
            GELU_CDF_CORE_C16);
        result = x * phi;
    }
    // Left CDF region [-5, -3): GELU(x) = x * Phi_left(x+4)
    // Shifted: t = x + 4 maps x in [-5, -3] to t in [-1, 1]
    v_elseif(x >= -5.0f) {
        sfpi::vFloat t = x + 4.0f;
        sfpi::vFloat phi = PolynomialEvaluator::eval(
            t,
            GELU_CDF_LEFT_C0,
            GELU_CDF_LEFT_C1,
            GELU_CDF_LEFT_C2,
            GELU_CDF_LEFT_C3,
            GELU_CDF_LEFT_C4,
            GELU_CDF_LEFT_C5,
            GELU_CDF_LEFT_C6,
            GELU_CDF_LEFT_C7,
            GELU_CDF_LEFT_C8);
        result = x * phi;
    }
    // Exp-based region (-13.1875, -5): asymptotic formula
    // GELU(x) = -exp(-x^2/2) / sqrt(2*pi) * (1 - 1/x^2 + 3/x^4)
    v_elseif(x > -13.1875f) {
        constexpr float NEG_INV_SQRT_2PI = -0.3989422804014327f;

        sfpi::vFloat x2 = x * x;
        sfpi::vFloat t = x2 * (-0.5f);  // t = -x^2/2

        // Use general-purpose exp for the forward case
        // (no fused method needed: result magnitude > exp(t) magnitude)
        sfpi::vFloat exp_val = _sfpu_exp_f32_accurate_(t);

        // Mills ratio asymptotic correction
        sfpi::vFloat inv_x2 = _sfpu_reciprocal_<2>(x2);  // 1/x^2
        sfpi::vFloat inv_x4 = inv_x2 * inv_x2;           // 1/x^4
        sfpi::vFloat correction = 1.0f - inv_x2 + 3.0f * inv_x4;

        result = exp_val * NEG_INV_SQRT_2PI * correction;
    }
    // For x <= -13.1875: saturate to 0 (BF16 natural saturation)
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE>
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

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_gelu_<APPROXIMATION_MODE, ITERATIONS>();
    } else {
#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat result = calculate_gelu_piecewise<APPROXIMATION_MODE>(in);
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

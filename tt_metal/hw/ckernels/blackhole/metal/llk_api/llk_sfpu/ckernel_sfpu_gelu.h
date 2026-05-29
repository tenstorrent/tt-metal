// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"  // For _sfpu_round_to_nearest_int32_
#include "sfpu/ckernel_sfpu_polyval.h"
#include "sfpu/ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_cdf.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "sfpi.h"

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
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

    // Step 2: Extended precision range reduction (Cody-Waite)
    sfpi::vFloat r = k * LN2_HI + t;
    r = k * LN2_LO + r;

    // Step 3: Polynomial approximation of exp(r) for |r| < 0.5
    constexpr float C2 = 0.5f;
    constexpr float C3 = 0.166666667f;
    constexpr float C4 = 0.0416666667f;
    constexpr float C5 = 0.00833333333f;
    sfpi::vFloat poly = PolynomialEvaluator::eval(r, 1.0f, 1.0f, C2, C3, C4, C5);

    // Step 4: FUSED MULTIPLY - multiply by x BEFORE exponent shift!
    // This is the key to avoiding intermediate underflow.
    // x * exp(t) = x * 2^k * poly = (x * poly) * 2^k
    sfpi::vFloat x_poly = x * poly;  // x * poly ≈ -9 to -13 (safe range)

    // Step 5: Exponent bit manipulation on the FUSED result
    sfpi::vInt xpoly_exp = sfpi::exexp(x_poly, sfpi::ExponentMode::NoDebias);  // Extract exponent of x*poly
    sfpi::vInt new_exp = xpoly_exp + k_int;                                    // Shift by 2^k

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

        // Ensure result has the same sign as input using copysgn
        result = copysgn(result, val);
    }
    v_endif;

    return result;
}

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
sfpi_inline sfpi::vFloat calculate_gelu_piecewise(sfpi::vFloat x) {
    sfpi::vFloat result = sfpi::vConst0;  // Default: 0 for x <= -13.1875
    sfpi::vFloat x2 = x * x;              // Hoisted: used by both core CDF and exp regions

    // v_if/v_and narrowing pattern: start with widest region, progressively
    // narrow the mask. Each v_and overwrites the result for the narrowed set.
    // This avoids separate branch mask setup per region (v_elseif).
    v_if(x > -13.1875f) {
        // Exp-based region (-13.1875, -3.125): exp(-x²/2) · P(x)
        // P(x) is a degree-4 correction polynomial fitted to the TRUE function
        // GELU(x)·exp(x²/2) via minimax approximation over x ∈ (-13.1875, -3.125).
        //
        // Uses Moroz exp_21f instead of Cody-Waite range reduction.
        // The intermediate t = -x²/2 is folded into the log2 constant,
        // computing x2 * (-0.5/ln2) + 127 directly (saves 1 mul).
        // Moroz exp_21f: compact exp via integer bit tricks
        // Fold t = -x²/2 into the log2 constant: x2 * (-0.5 / ln2) + 127
        constexpr float NEG_HALF_ONE_LN2 = -0.72134752044f;  // -0.5 / ln(2)
        sfpi::vFloat xlog2 = x2 * NEG_HALF_ONE_LN2 + 127.0f;

        sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);
        sfpi::vInt exponential_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z), sfpi::ExponentMode::NoDebias);
        sfpi::vInt fractional_part = sfpi::exman(sfpi::reinterpret<sfpi::vFloat>(z));

        sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, sfpi::RoundMode::NearestEven);
        frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);
        sfpi::vFloat exp_val = sfpi::setexp(frac, exponential_part);

        // Correction polynomial: P(x) ≈ GELU(x)·exp(x²/2)
        sfpi::vFloat correction = PolynomialEvaluator::eval(
            x, GELU_EXP_CORR_C0, GELU_EXP_CORR_C1, GELU_EXP_CORR_C2, GELU_EXP_CORR_C3, GELU_EXP_CORR_C4);

        result = exp_val * correction;

        // Core CDF region [-3.125, 2.78125): GELU(x) = x * Phi_core(x)
        // Phi(x) = C0 + x*(C1 + x^2*(C3 + x^2*(C5 + ... + x^2*C15)))
        // Factored via u=x^2 to eliminate zero even-power coefficients
        v_and(x >= -3.125f);
        sfpi::vFloat odd_poly = PolynomialEvaluator::eval(
            x2,
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

        // Identity region: x >= 2.78125
        v_and(x >= 2.78125f);
        result = x;
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void gelu_init() {
    if constexpr (APPROXIMATION_MODE) {
        sfpi::vConstFloatPrgm0 = 0.5f;

        // // >= 3.0f
        // lreg2_hi=0.50;//3800
        // lreg6_hi=0.0f;//7c00
        // // 2.0f -> 3.0f
        // lreg2_lo= 0.5402f;//3852
        // lreg6_lo= -0.1194f;//AFA4
        // // 1.5f -> 2.0f
        // lreg1_hi= .6099f; //38E1
        // lreg5_hi= -.2635f; //B437
        // // 1.0f -> 1.5f
        // lreg1_lo=0.6189;//38F3
        // lreg5_lo=-.2797;//B479
        // // 0.5f -> 1.0f
        // lreg0_hi=.4939f;//37E7
        // lreg4_hi=-.1605f;//B122
        // // 0.0f -> 0.5f
        // lreg0_lo=0.1928f;//322B
        // lreg4_lo=-0.0150f;//A3AE
        _sfpu_load_imm32_(0, 0x37E7322B);
        _sfpu_load_imm32_(4, 0xB12286D8);

        _sfpu_load_imm32_(1, 0x38E138F3);
        _sfpu_load_imm32_(5, 0xB437B479);

        _sfpu_load_imm32_(2, 0x38003852);
        _sfpu_load_imm32_(6, 0x7c00afa4);
    }
    // Accurate mode: no init needed (correction polynomial replaces reciprocal)
}

template <bool APPROXIMATION_MODE>
void gelu_derivative_init() {
    std::uint32_t imm0;
    std::uint32_t imm1;
    std::uint32_t imm2;
    std::uint32_t imm3;
    std::uint32_t imm4;
    std::uint32_t imm5;

    if constexpr (APPROXIMATION_MODE) {
        // Using a 6 piece LUT to calculate and model gelu_derivative directly
        // x <= 0.5 --> 0.8x + 0.5
        // x <= 1.0 --> 0.4x + 0.7
        // x <= 1.5 --> 0.1x + 0.99
        // x <= 2.0 --> -0.09x + 1.27
        // x <= 3.0 --> -0.075x + 1.235
        // x >  3.0 --> 1.0
        // imm0[15:0] = A0=0.8    = 0x3A66 -- imm0[31:16] = A1=0.4   = 0x3666
        imm0 = 0x36663A66;
        // imm1[15:0] = A2=0.1    = 0x2E66 -- imm1[31:16] = A3=-0.09 = 0xADC3
        imm1 = 0xADC32E66;
        // imm2[15:0] = A4=-0.075 = 0xACCD -- imm2[31:16] = A5=0     = 0x7C00
        imm2 = 0x7C00ACCD;
        // imm3[15:0] = B0=0.5    = 0x3800 -- imm3[31:16] = B1=0.7   = 0x399A
        imm3 = 0x399A3800;
        // imm4[15:0] = B2=0.99   = 0x3BEC -- imm4[31:16] = B3=1.27  = 0x3D14
        imm4 = 0x3D143BEC;
        // imm5[15:0] = B4=1.235  = 0x3CF1 -- imm5[31:16] = B5=1.0   = 0x3C00
        imm5 = 0x3C003CF1;
        _sfpu_load_imm32_(0, imm0);
        _sfpu_load_imm32_(1, imm1);
        _sfpu_load_imm32_(2, imm2);
        _sfpu_load_imm32_(4, imm3);
        _sfpu_load_imm32_(5, imm4);
        _sfpu_load_imm32_(6, imm5);
    } else {
        // Initialisation for use of _calculate_exponential_body_<false>.
        exp_init<false, 0x3F800000>();

        imm0 = 0x28FF;
        imm1 = 0x3020;
        _sfpu_load_imm16_(0, imm0);
        _sfpu_load_imm16_(1, imm1);
    }
}

template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _calculate_gelu_core_(sfpi::vFloat in) {
    // SFPU microcode:
    // result = (APPROX_MODE == 1)
    //   ? (1 + erf(x/sqrt(2)))
    //   : (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) )
    sfpi::vFloat result;
    if constexpr (APPROXIMATION_MODE) {
        result = in;
    } else {
        // f = (0.044715*x^3 + x)
        result = (in * in) * (in * sfpi::sFloat16b(0.044715f)) + in;
        result *= sfpi::sFloat16b(0.79788f);
    }

    return result;
}

template <int ITERATIONS>
inline void _calculate_gelu_appx_() {
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // sfpi::vFloat in = sfpi::dst_reg[0];
        // sfpi::vFloat result = calculate_gelu_core<APPROXIMATION_MODE>(in);

        // sfpi::vFloat half_in = in * half;
        // result = lut(result, l0, l1, l2);
        // result = half_in * result + half_in;

        // sfpi::dst_reg[0] = result;

        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat half = sfpi::vConstFloatPrgm0;
        sfpi::vFloat half_in = in * half;
        sfpi::vFloat result = lut2_sign(in, l0, l1, l2, l4, l5, l6);
        result = half_in + result;

        sfpi::dst_reg[0] = result;

        sfpi::dst_reg++;

        // sfpi::dst_reg++;
        // TTI_SFPLOAD(3, 0, 1/*load addr mode*/,0);    // load from dest
        ////TTI_SFPMUL(3,11,9,7,0);           // lreg7 = 0.5*lreg3
        // TTI_SFPLUTFP32(7, 2);                // lreg7= LUT(3)
        // TTI_SFPMAD(3,12,7,3,0);            // lreg3 = 0.5*lreg3+lregm7
        // TTI_SFPSTORE(3, 0, 3/*store_addr_mod3*/, 0);   // and INCRWC by 4 using mode 3
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
    sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
    sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
    sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
}

template <int ITERATIONS>
inline void _calculate_gelu_accurate_() {
    constexpr bool scaled = true;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = _calculate_cdf_appx_(in, scaled);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_gelu_appx_<ITERATIONS>();
    } else {
#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat result = calculate_gelu_piecewise(in);
            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::NearestEven);
            }
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu_derivative() {
    if constexpr (APPROXIMATION_MODE) {
        constexpr int lut_mode = 1;  // SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1

        sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
        sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
        sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
        sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
        sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
        sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];

// SFPU microcode:
#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            val = lut2(val, l0, l1, l2, l4, l5, l6, lut_mode);
            v_if(val < 0.0F) { val = val + 1.0f; }
            v_endif;
            sfpi::dst_reg[0] = val;
            sfpi::dst_reg++;
        }

        sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
        sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
        sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
        sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
        sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
        sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
    } else {
        constexpr std::uint32_t imm2 = 0xFF10;

        sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
        sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];

// SFPU microcode:
#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat neg_half_sq_in = in * in * -0.5f;

            // exp = e^(val)
            sfpi::vFloat exp = _calculate_exponential_body_<false>(neg_half_sq_in);

            // exp = exp * 1/sqrt(2*pi)
            sfpi::vFloat partial = exp * in * sfpi::sFloat16b(0.3989423F);

            sfpi::vFloat result = _calculate_gelu_core_<true>(in);

            result = lut(result, l0, l1, imm2);

            sfpi::dst_reg[0] = partial + result + 0.5f;
            sfpi::dst_reg++;
        }

        sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
        sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    }
}

// =============================================================================
// GELU Derivative - Polynomial Approximation
// =============================================================================
// GELU'(x) = Φ(x) + x*φ(x) where Φ is CDF, φ is PDF of standard normal
//
// Uses the identity GELU'(x) + GELU'(-x) = 1 (provable from Φ(x) + Φ(-x) = 1
// and φ(x) = φ(-x)), so GELU'(x) - 0.5 is an odd function of x, expressible as
// x * h(x²) for some function h. This halves the polynomial degree.
//
// Three active regions (plus zero default):
//   x >= 3.1719:          Saturation to 1
//   -3 <= x < 3.1719:     Core: 0.5 + x * h(x²), degree-8 in u=x²
//   -13.375 < x < -3:     Asymptotic: x * exp(-x²/2) with Mills ratio correction
//   x <= -13.375:          Zero (BF16 natural saturation)
// =============================================================================

// Degree-8 polynomial h(u) for GELU'(x) = 0.5 + x * h(x²) over [-3, 3.1719]
// Exploits odd-function decomposition: only 9 coefficients instead of 17.
// Evaluation: 1 MUL (u=x²) + 9 MAD (Horner on u) + 1 MUL (x*h) + 1 ADD (+0.5)
// = ~12 ops, vs ~32 ops for degree-16 in x.
// Coefficients from Sollya fpminimax.
constexpr float GELU_DERIV_H0 = 7.9788309336e-01f;
constexpr float GELU_DERIV_H1 = -2.6593554020e-01f;
constexpr float GELU_DERIV_H2 = 5.9766173363e-02f;
constexpr float GELU_DERIV_H3 = -9.4142099842e-03f;
constexpr float GELU_DERIV_H4 = 1.1061388068e-03f;
constexpr float GELU_DERIV_H5 = -9.7448595625e-05f;
constexpr float GELU_DERIV_H6 = 6.0921474869e-06f;
constexpr float GELU_DERIV_H7 = -2.3764030743e-07f;
constexpr float GELU_DERIV_H8 = 4.2734988881e-09f;

// GELU Derivative Evaluation with Polynomial Approximation
// Saturation thresholds derived from exhaustive BF16 research (DAZ+FTZ model):
// - Zero saturation: x <= -13.375 (GELU'(x) becomes 0 in BF16)
// - One saturation: x >= 3.1719 (GELU'(x) becomes exactly 1 in BF16)
// Note: GELU'(x) has a "hump" exceeding 1.0 for x in [0.77, 3.16]
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_gelu_derivative_simple(sfpi::vFloat x) {
    sfpi::vFloat result = sfpi::vConst0;  // Default: 0 for x <= -13.375

    // For x >= 3.1719, output saturates to 1 (verified saturation threshold)
    v_if(x >= 3.1719f) { result = sfpi::vConst1; }
    // Core region [-3, 3.1719]: GELU'(x) = 0.5 + x * h(x²)
    // Odd-function decomposition: GELU'(x) + GELU'(-x) = 1, so GELU'(x) - 0.5
    // is odd and can be written as x * h(x²). Degree-8 in u=x² (~12 ops vs ~32).
    v_elseif(x >= -3.0f) {
        sfpi::vFloat u = x * x;
        sfpi::vFloat h = PolynomialEvaluator::eval(
            u,
            GELU_DERIV_H0,
            GELU_DERIV_H1,
            GELU_DERIV_H2,
            GELU_DERIV_H3,
            GELU_DERIV_H4,
            GELU_DERIV_H5,
            GELU_DERIV_H6,
            GELU_DERIV_H7,
            GELU_DERIV_H8);
        result = 0.5f + x * h;
    }
    // Tail region (-13.375, -3): asymptotic formula with fused x*exp(t)
    // LEFT polynomial eliminated — the asymptotic formula achieves Max ULP ≤ 1
    // across the entire (-13.375, -3) range.
    // GELU'(x) ≈ φ(x) * (x - 1/x + 1/x³) where φ(x) = exp(-x²/2) / sqrt(2π)
    //
    // Uses x_times_exp_negative_tail() for fused x * exp(t) to avoid intermediate
    // underflow. Cody-Waite range reduction + degree-5 Taylor + direct exponent
    // bit manipulation.
    //
    // For APPROXIMATION_MODE=true: use simple x*φ(x) (~1% relative error)
    // For APPROXIMATION_MODE=false: use Mills ratio correction (<0.01% relative error)
    v_elseif(x > -13.375f) {
        constexpr float INV_SQRT_2PI = 0.3989422804014327f;  // 1/sqrt(2*pi)

        sfpi::vFloat x2 = x * x;
        sfpi::vFloat t = x2 * (-0.5f);  // t = -x²/2

        sfpi::vFloat x_exp = x_times_exp_negative_tail(x, t);

        if constexpr (APPROXIMATION_MODE) {
            result = x_exp * INV_SQRT_2PI;
        } else {
            sfpi::vFloat inv_x2 = _sfpu_reciprocal_<2>(x2);  // 1/x²
            sfpi::vFloat inv_x4 = inv_x2 * inv_x2;           // 1/x⁴
            sfpi::vFloat correction = 1.0f - inv_x2 + inv_x4;
            result = x_exp * INV_SQRT_2PI * correction;
        }
    }
    // For x <= -13.375, saturate to 0
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_gelu_derivative_polynomial() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = calculate_gelu_derivative_simple<APPROXIMATION_MODE>(val);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::NearestEven);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void gelu_derivative_polynomial_init() {
    if constexpr (!APPROXIMATION_MODE) {
        // Call _init_sfpu_reciprocal_ directly: gelu derivative uses _sfpu_reciprocal_<2>
        // inline (not _calculate_reciprocal_internal_), so SFPLOADMACRO fast-path init is
        // not needed. On BH, _init_reciprocal_ omits _init_sfpu_reciprocal_ (it only
        // configures SFPLOADMACRO macros), so vConstFloatPrgm0=2.0f would be unset.
        _init_sfpu_reciprocal_<false>();
    }
}

}  // namespace sfpu
}  // namespace ckernel

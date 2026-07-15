// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"

#include "ckernel_sfpu_exp.h"  // For _sfpu_round_to_nearest_int32_
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_cdf.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "ckernel_sfpu_erf.h"  // ERF_LUT, ERF_NUM_DEGREE, ERF_DEN_DEGREE (INP_FLOAT32 branch for FP32 path)
#include "ckernel_sfpu_piecewise_rational.h"
#include "ckernel_sfpu_tanh.h"  // _sfpu_tanh_fp32_accurate_ for gelu_tanh
#include "sfpi.h"

namespace ckernel::sfpu {

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
    sfpi::vInt xpoly_exp = sfpi::exexp(x_poly, sfpi::ExponentMode::Biased);  // Extract exponent of x*poly
    sfpi::vInt new_exp = xpoly_exp + k_int;                                  // Shift by 2^k

    // Step 6: FTZ check on FINAL result (x * exp(t)), not intermediate exp(t)
    sfpi::vFloat result = 0.0f;
    v_if(new_exp > 0) { result = sfpi::setexp(x_poly, new_exp); }
    v_endif;

    return result;
}

// =============================================================================
// Forward GELU - Piecewise CDF Approximation
// =============================================================================
// GELU(x) = x * Phi(x) where Phi(x) = 0.5*(1+erf(x/sqrt(2))) is the CDF
//
// Three active regions (plus zero default):
//   x >= 2.78125:                  Identity (result = x)
//   -3.125 <= x < 2.78125:         Core CDF polynomial (degree-13 in u=x²)
//   -5.54259443 < x < -3.125:      Moroz exp_21f: GELU = x · round_to_grid(H), H = exp(-x²/2)·corr_H
//   x <= -5.54259443:               Zero (matches torch.gelu BF16 saturation)
//
// Deep-tail grid rounding: in the deep tail (-5.54259443, -4.828125] torch computes
// GELU(x) = x · float32(½·erfc(|x|/√2)) where the float32 ½·erfc is a *staircase*
// (multiple BF16 inputs share one float32 bucket). Every bucket is an exact integer
// multiple of 2⁻²⁵ (k ∈ {1..21}). A smooth polynomial diverges from this staircase by
// up to 116 BF16 ULP. Instead we compute the smooth H and round it to the nearest multiple
// of 2⁻²⁵ before multiplying by x — reproducing torch's exact bucket at 2 ops
// (vs a 27-op step-function cascade). Above -4.828125 the 2⁻²⁵ grid is far finer than
// BF16 so the rounding is lossless, letting one rounded multiply serve the whole exp region.
// Result: ULP ≤ 1 across the tail (0 ULP for almost all of it), with DKD below main.
//
// v_if/v_and narrowing: widest region first, each v_and overwrites for narrower set.
// =============================================================================

// Degree-13 CDF polynomial for Phi(x) over [-3.125, 2.78125]
// Phi(x) = 0.5 + x * p(x²); only odd powers, evaluated via u=x² Horner (7 MADs).
// Minimax-fitted with BF16-ULP weighting → MaxULP = 0.87 < 1.
constexpr float GELU_CDF_CORE_C0 = 5.000000000e-01f;
constexpr float GELU_CDF_CORE_C1 = 3.9894227818e-01f;
constexpr float GELU_CDF_CORE_C3 = -6.6361041488e-02f;
constexpr float GELU_CDF_CORE_C5 = 9.7720050615e-03f;
constexpr float GELU_CDF_CORE_C7 = -1.0717806322e-03f;
constexpr float GELU_CDF_CORE_C9 = 8.1812159812e-05f;
constexpr float GELU_CDF_CORE_C11 = -3.8082057209e-06f;
constexpr float GELU_CDF_CORE_C13 = 7.9821413868e-08f;

// Degree-3 H-form correction polynomial for the exp-based region (-5.54259443, -3.125].
// H(x) = ½·erfc(|x|/√2) ≈ exp(-x²/2) · corr_H(x), and GELU(x) = x · H(x).
// Fitted (uniform relative error) so the deep-tail snap is exact and the smooth
// region stays ULP ≤ 1. See deep-tail snap below for why the H-form matters.
constexpr float GELU_HCORR_C0 = 3.0369991064e-01f;
constexpr float GELU_HCORR_C1 = 9.5413386822e-02f;
constexpr float GELU_HCORR_C2 = 1.3809983619e-02f;
constexpr float GELU_HCORR_C3 = 7.5950479368e-04f;

// Forward GELU Evaluation with CDF Polynomial Approximation
// GELU(x) = x * Phi(x) where Phi is approximated piecewise
sfpi_inline sfpi::vFloat calculate_gelu_piecewise(sfpi::vFloat x) {
    sfpi::vFloat result = 0.0f;  // Default: 0 for x <= -5.54259443 (torch saturation)
    sfpi::vFloat x2 = x * x;

    v_if(x > -5.54259443f) {
        // Shared smooth H ≈ ½·erfc(|x|/√2) via Moroz exp_21f · corr_H.
        constexpr float NEG_HALF_ONE_LN2 = -0.72134752044f;  // -0.5 / ln(2)
        sfpi::vFloat xlog2 = x2 * NEG_HALF_ONE_LN2 + 127.0f;

        sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);
        sfpi::vInt exponential_part = sfpi::exexp(sfpi::as<sfpi::vFloat>(z), sfpi::ExponentMode::Biased);
        sfpi::vMag fractional_part = sfpi::exman(sfpi::as<sfpi::vFloat>(z));

        sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
        frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);
        sfpi::vFloat exp_val = sfpi::setexp(frac, exponential_part);

        sfpi::vFloat H =
            exp_val * PolynomialEvaluator::eval(x, GELU_HCORR_C0, GELU_HCORR_C1, GELU_HCORR_C2, GELU_HCORR_C3);

        // Exp region (-5.54259443, -3.125]: GELU = x · round_to_grid(H).
        // Round H to the nearest multiple of 2⁻²⁵: adding 0.375 = (2²³+2²²)·2⁻²⁵ shifts H's
        // 2⁻²⁵ place to the FP32 round-to-nearest-even boundary (safe since H ∈ [3e-8, 9e-4] ≪ 0.25),
        // so (H + 0.375) - 0.375 == round(H / 2⁻²⁵) · 2⁻²⁵. In the deep tail (-5.5426, -4.828]
        // this reproduces torch's exact float32 ercc staircase (0 BF16 ULP); above -4.828 the
        // 2⁻²⁵ grid is far finer than BF16 so rounding is lossless. One constant serves the
        // whole exp region — no separate branch or multiply.
        constexpr float ROUND_TO_GRID = 0.375f;
        sfpi::vFloat Hs = (H + ROUND_TO_GRID) - ROUND_TO_GRID;
        result = x * Hs;

        // Core CDF region [-3.125, 2.78125): GELU = x · Phi_core(x).
        v_and(x >= -3.125f);
        sfpi::vFloat odd_poly = PolynomialEvaluator::eval(
            x2,
            GELU_CDF_CORE_C1,
            GELU_CDF_CORE_C3,
            GELU_CDF_CORE_C5,
            GELU_CDF_CORE_C7,
            GELU_CDF_CORE_C9,
            GELU_CDF_CORE_C11,
            GELU_CDF_CORE_C13);
        sfpi::vFloat phi = GELU_CDF_CORE_C0 + x * odd_poly;
        result = x * phi;

        // Identity region x >= 2.78125.
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

        // LUT segments (6-entry piecewise linear, each hi/lo pair packed into one imm32):
        // [0.0, 0.5): slope=0.1928, intercept=-0.0150  (lreg0)
        // [0.5, 1.0): slope=0.4939, intercept=-0.1605  (lreg0 hi / lreg4 hi)
        // [1.0, 1.5): slope=0.6189, intercept=-0.2797  (lreg1)
        // [1.5, 2.0): slope=0.6099, intercept=-0.2635  (lreg1 hi / lreg5 hi)
        // [2.0, 3.0): slope=0.5402, intercept=-0.1194  (lreg2)
        // [3.0, ∞):   slope=0.5,    intercept=0.0      (lreg2 hi / lreg6 hi)
        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(0x37E7322B);
        sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vUInt(0xB12286D8);

        sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(0x38E138F3);
        sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vUInt(0xB437B479);

        sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(0x38003852);
        sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vUInt(0x7c00afa4);
    } else if constexpr (is_fp32_dest_acc_en) {
        // FP32 accurate mode: rational erf evaluation requires reciprocal init
        sfpu_reciprocal_init<false>();
    }
    // BF16 accurate mode: no init needed (correction polynomial has no reciprocal)
}

template <int ITERATIONS>
inline void calculate_gelu_appx() {
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat half = sfpi::vConstFloatPrgm0;
        sfpi::vFloat half_in = in * half;
        sfpi::vFloat result = lut2_sign(in, l0, l1, l2, l4, l5, l6);
        result = half_in + result;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
    sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
    sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
    sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
}

// FP32 erf: n16/d16 parity rational coefficients, MaxULP=1 vs FP64.
// Split from ERF_LUT (ckernel_sfpu_erf.h INP_FLOAT32 branch) entries [2..18] (num) and [19..35] (den).
// piecewise_rational_eval_parity_numer_denom() takes const float* — no std::array needed.
constexpr float GELU_ERF_NUM[17] = {  // odd powers only (c0=0, c2=0, ..., c16=0)
    0.0f,
    1.1283791065e+00f,
    0.0f,
    2.1477432549e-01f,
    0.0f,
    6.2133435160e-02f,
    0.0f,
    5.6230435148e-03f,
    0.0f,
    6.1307044234e-04f,
    0.0f,
    1.7678321456e-05f,
    0.0f,
    2.7384647439e-08f,
    0.0f,
    -2.8632063387e-10f,
    0.0f};
constexpr float GELU_ERF_DEN[17] = {  // even powers only (c1=0, c3=0, ..., c15=0)
    1.0f,
    0.0f,
    5.2367275953e-01f,
    0.0f,
    1.2961706519e-01f,
    0.0f,
    1.9642570987e-02f,
    0.0f,
    1.9545555115e-03f,
    0.0f,
    1.3179056987e-04f,
    0.0f,
    1.3156344494e-06f,
    0.0f,
    -3.5153888689e-09f,
    0.0f,
    -6.7350725691e-12f};

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        calculate_gelu_appx<ITERATIONS>();
    } else if constexpr (is_fp32_dest_acc_en) {
        // FP32 accurate mode: GELU(x) = x * 0.5 * (1 + erf(x/√2))
        // using the n16/d16 piecewise rational erf (MaxULP=1 vs FP64).
        // unroll 8 fills the SFPU pipeline across 8 independent dst-tile chains.
        constexpr float INV_SQRT2 = 0.7071067811865475f;
        constexpr float GELU_SAT = -5.54259443f;  // 0xc0b15cef: first x where libm erff=-1.0
#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];
            sfpi::vFloat result = 0.0f;  // default 0 for x <= GELU_SAT
            v_if(x > GELU_SAT) {
                sfpi::vFloat scaled = x * INV_SQRT2;
                scaled = sfpi::min(scaled, 10.0f);
                sfpi::vFloat x2 = scaled * scaled;
                sfpi::vFloat erf_n, erf_d;
                piecewise_rational_eval_parity_numer_denom<16, 16>(
                    GELU_ERF_NUM, GELU_ERF_DEN, scaled, x2, erf_n, erf_d);
                sfpi::vFloat erf_val = erf_n * sfpu_reciprocal<false>(erf_d);
                erf_val = sfpi::clamp(erf_val, -1.0f, 1.0f);
                result = x * (0.5f + 0.5f * erf_val);
                // Stuck-erff guard: when rational rounds erf to -1.0 inside computation zone,
                // glibc/torch overestimates erfc → first stuck level = x * 2^-25.
                constexpr float HALF_ULP_AT_1 = 2.9802322387695313e-08f;  // 2^-25
                v_if(erf_val == -1.0f) { result = x * HALF_ULP_AT_1; }
                v_endif;
            }
            v_endif;
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else {
        // BF16 accurate mode: piecewise CDF with Max ULP=1 vs true GELU.
        // unroll 8 fills the SFPU pipeline across 8 independent dst-tile chains.
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat result = calculate_gelu_piecewise(in);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}

// =============================================================================
// GELU tanh approximation in FP32:
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// =============================================================================

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_gelu_tanh() {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float GELU_TANH_K = 0.044715f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        sfpi::vFloat x2 = x * x;
        sfpi::vFloat p = GELU_TANH_K * x2 + 1.0f;
        sfpi::vFloat q = x * p;
        sfpi::vFloat u = SQRT_2_OVER_PI * q;
        sfpi::vFloat t = _sfpu_tanh_fp32_accurate_(u);

        // reload due to register pressure
        x = sfpi::dst_reg[0];
        sfpi::vFloat result = sfpi::copysgn(sfpi::vFloat(0.0f), x);

        sfpi::vFloat half_x = 0.5f * x;
        result = half_x * t + half_x;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool is_fp32_dest_acc_en>
inline void gelu_tanh_init() {
    // initialise constants for _sfpu_tanh_fp32_accurate_
    tanh_init<false, true>();
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
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_gelu_derivative_simple(sfpi::vFloat x) {
    sfpi::vFloat result = 0.0f;  // Default: 0 for x <= -13.375

    // For x >= 3.1719, output saturates to 1 (verified saturation threshold)
    v_if(x >= 3.1719f) { result = 1.0f; }
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
            // 1 NR step suffices for BF16 (7 mantissa bits); FP32 needs 2 steps (23 bits).
            sfpi::vFloat inv_x2 = sfpu_reciprocal_iter<is_fp32_dest_acc_en ? 2 : 1>(x2);  // 1/x²
            sfpi::vFloat inv_x4 = inv_x2 * inv_x2;                                        // 1/x⁴
            sfpi::vFloat correction = 1.0f - inv_x2 + inv_x4;
            result = x_exp * INV_SQRT_2PI * correction;
        }
    }
    // For x <= -13.375, saturate to 0
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_gelu_derivative_polynomial() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = calculate_gelu_derivative_simple<APPROXIMATION_MODE, is_fp32_dest_acc_en>(val);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void gelu_derivative_polynomial_init() {
    if constexpr (!APPROXIMATION_MODE) {
        // Call sfpu_reciprocal_init directly: gelu derivative uses sfpu_reciprocal_iter
        // inline (not _calculate_reciprocal_internal_), so SFPLOADMACRO fast-path init is
        // not needed. On BH, _init_reciprocal_ omits sfpu_reciprocal_init (it only
        // configures SFPLOADMACRO macros), so vConstFloatPrgm0=2.0f would be unset.
        //
        // Not keyed on is_fp32_dest_acc_en: the template arg to sfpu_reciprocal_iter<N>
        // selects the Newton-Raphson step count (2 for FP32, 1 for BF16), but
        // sfpu_reciprocal_init only seeds the initial estimate and does not vary
        // with N. A single <false> call therefore covers both precisions. The
        // asymmetry vs gelu_init() (which gained an fp32-specific branch) is
        // intentional — the derivative path has no LUT to load.
        sfpu_reciprocal_init<false>();
    }
}

}  // namespace ckernel::sfpu

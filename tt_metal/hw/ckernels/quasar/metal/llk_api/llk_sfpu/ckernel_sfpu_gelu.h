// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_exp.h"  // _float_to_int32_for_exp_21f_ (BF16 CDF path)
#include "ckernel_sfpu_piecewise_rational.h"
#include "ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// =============================================================================
// Approximate GELU - 6-segment piecewise-linear LUT
// =============================================================================
// GELU(x) ~= 0.5 * x + f(|x|), with f evaluated by the SFPU LUT unit. Same table and same math as
// Blackhole's calculate_gelu_appx (gelu_init has the segment breakdown); only the table location
// differs. Quasar's SFPLUTFP32 reads the table from the constant LRegs 9-14 and the looked-up value
// from LReg3, where WH/BH read the table from LReg0-2/LReg4-6.
//
// So gelu_init loads the table with _sfpu_load_config32_ rather than sfpi: vCReg assignment reaches
// SFPCONFIG, but the assembler rejects dests 9 and 10 ("permitted mask is 0xf9ff"), which is two of
// the six table words. See tt-metal issue #51346.
//
// The loop below is still plain sfpi. lut2_sign() wants six LReg operands because that is where WH/BH
// keep the table, and it pins the looked-up value to LReg3, which is what this hardware wants anyway.
// The LUT ignores those six registers here, so reading them back through l_reg[] (no instructions
// emitted) is enough to satisfy the intrinsic. Loading the table into them, the way Blackhole's init
// does, is what miscompares.
//
// Two things to avoid: vConstFloatPrgm0 for the 0.5 (that is LReg12, a table intercept word; an
// SFPMULI immediate is free), and any sfpi code between this init and the loop, since the table
// overwrites the SFPU constants 0.0/1.0/-1.0 in LReg9/10/11. Each op's init restores them via
// _init_sfpu_config_reg_, so a following op is unaffected.
// =============================================================================

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_gelu_appx() {
    // Operands lut2_sign() requires; their contents are irrelevant on Quasar (see above).
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat piecewise = lut2_sign(in, l0, l1, l2, l4, l5, l6);
        sfpi::dst_reg[0] = in * 0.5f + piecewise;
        sfpi::dst_reg++;
    }
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
        // this reproduces torch's exact float32 erfc staircase (0 BF16 ULP); above -4.828 the
        // 2⁻²⁵ grid is far finer than BF16 so rounding is lossless. One constant serves the
        // whole exp region — no separate branch or multiply.
        constexpr float ROUND_TO_GRID = 0.375f;
        sfpi::vFloat Hs = (H + ROUND_TO_GRID) - ROUND_TO_GRID;
        result = x * Hs;

        // Core CDF region (-3.125, 2.78125]: GELU = x · Phi_core(x).
        // Blackhole narrows with >=. Strict > here because a float >= is unreliable at the boundary
        // itself on Quasar (issue #50208), and each fit is closed at its own end anyway: the H-form
        // covers (-5.54259443, -3.125] and the core polynomial [-3.125, 2.78125].
        v_and(x > -3.125f);
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

        // Identity region x > 2.78125.
        v_and(x > 2.78125f);
        result = x;
    }
    v_endif;

    return result;
}

/**
 * @brief Program the constants the selected GELU implementation reads.
 *
 * @tparam APPROXIMATION_MODE: Select the LUT path (loads the 6-segment table), values = <true/false>
 * @tparam is_fp32_dest_acc_en: For the non-approximate path, select the fp32 rational-erf variant,
 *         which needs the Newton-Raphson constant its reciprocal refines with.
 * @note The surrounding @ref llk_math_eltwise_unary_sfpu_init resets the counters, so unlike the
 *       Blackhole kernel this init only programs constants.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void gelu_init() {
    if constexpr (APPROXIMATION_MODE) {
        // Segment slopes/intercepts, packed hi/lo the way SFPLUTFP32's FP16 6-entry mode 1 reads
        // them. Same values as the Blackhole table (l_reg0 = 0x37E7322B, l_reg4 = 0xB12286D8, ...).
        //
        // 1.0 > x >= 0.5
        // lreg9_hi  =  0.4939 (0x37E7)
        // lreg12_hi = -0.1605 (0xB122)
        // 0.5 > x >= 0.0
        // lreg9_lo  =  0.1928 (0x322B)
        // lreg12_lo = -7.4e-05 (0x86D8)
        ckernel::math::_sfpu_load_config32_(0x9, 0x37E7, 0x322B);
        ckernel::math::_sfpu_load_config32_(0xC, 0xB122, 0x86D8);

        // 2.0 > x >= 1.5
        // lreg10_hi =  0.6099 (0x38E1)
        // lreg13_hi = -0.2635 (0xB437)
        // 1.5 > x >= 1.0
        // lreg10_lo =  0.6189 (0x38F3)
        // lreg13_lo = -0.2797 (0xB479)
        ckernel::math::_sfpu_load_config32_(0xA, 0x38E1, 0x38F3);
        ckernel::math::_sfpu_load_config32_(0xD, 0xB437, 0xB479);

        // x >= 3.0
        // lreg11_hi =  0.50   (0x3800)
        // lreg14_hi =  0.0    (0x7C00)
        // 3.0 > x >= 2.0
        // lreg11_lo =  0.5402 (0x3852)
        // lreg14_lo = -0.1194 (0xAFA4)
        ckernel::math::_sfpu_load_config32_(0xB, 0x3800, 0x3852);
        ckernel::math::_sfpu_load_config32_(0xE, 0x7C00, 0xAFA4);
    } else if constexpr (is_fp32_dest_acc_en) {
        // FP32 accurate mode: rational erf evaluation requires reciprocal init
        _init_reciprocal_<false>();
    }
    // BF16 accurate mode: no init needed (correction polynomial has no reciprocal)
}

// FP32 erf: n16/d16 parity rational coefficients, MaxULP=1 vs FP64.
// Split from Blackhole's ERF_LUT (ckernel_sfpu_erf.h INP_FLOAT32 branch) entries [2..18] (num) and
// [19..35] (den). piecewise_rational_eval_parity_numer_denom() takes const float* — no std::array needed.
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

/**
 * @brief Compute GELU in-place over a Dest tile.
 *
 * @tparam APPROXIMATION_MODE: Select the LUT path, values = <true/false>
 * @tparam is_fp32_dest_acc_en: 32-bit Dest; selects the fp32 rational-erf path and skips the
 *         round-to-nearest-bf16 conversion the 16-bit Dest store needs.
 * @tparam ITERATIONS: Number of SFPU loop iterations over the Dest face.
 * @note Call @ref gelu_init with matching template args first.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        calculate_gelu_appx<ITERATIONS>();
    } else if constexpr (is_fp32_dest_acc_en) {
        // FP32 accurate mode: GELU(x) = x * 0.5 * (1 + erf(x/√2))
        // using the n16/d16 piecewise rational erf (MaxULP=1 vs FP64).
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
                sfpi::vFloat erf_val = erf_n * _sfpu_reciprocal_<2>(erf_d);
                // Blackhole clamps erf_val to [-1, 1] and then guards the stuck-erff level with
                //   v_if (erf_val == -1.0f) { result = x * 2^-25; }
                // (where the rational rounds erf to -1.0, glibc/torch overestimate erfc, and the first
                // stuck level is x * 2^-25). Both fold into min/max on phi = 0.5 + 0.5*erf_val: the
                // same clamp on [0, 1], with the 2^-25 floor standing in for the guard.
                //
                // The clamp is what forced this, not the guard. sfpi's min/max/clamp lower to SFPSWAP
                // without the bit that selects a float compare, so Quasar compares the raw words as
                // two's complement. That is correct unless both operands are negative, which is
                // exactly the erf_val vs -1.0 case: it returned -1.0 for every erf_val in (-1, 0) and
                // collapsed every negative-input lane to +-0. Bounding phi keeps both operands of each
                // min/max non-negative.
                constexpr float HALF_ULP_AT_1 = 2.9802322387695313e-08f;  // 2^-25
                sfpi::vFloat phi = 0.5f + 0.5f * erf_val;
                phi = sfpi::max(phi, HALF_ULP_AT_1);  // erf_val <= -1, incl. the stuck level
                phi = sfpi::min(phi, 1.0f);           // erf_val >= 1
                result = x * phi;
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

}  // namespace ckernel::sfpu

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
// GELU(x) ~= 0.5 * x + f(|x|), f from the SFPU LUT unit; same table and math as Blackhole.
// gelu_init loads the table into the LUT config registers, overwriting the SFPU constants in
// LReg9-12, so no sfpi code that needs them may run between the init and the loop.
//
// Passes are pipelined (4 slots each): the next pass's load and the previous pass's store sit
// between the MAD and the LUT. The store advances Dest through ADDR_MOD_6, set by gelu_init<true>.
// Values carried between passes are pinned through l_reg[] so the recorded body ends in the state it
// starts in. Quasar cannot execute while recording, so the body is recorded, then replayed per pass.
// =============================================================================

inline constexpr sfpi::LRegs GELU_APPX_HALF_LREG = sfpi::LRegs::LReg0;
inline constexpr sfpi::LRegs GELU_APPX_RESULT_LREG = sfpi::LRegs::LReg1;
inline constexpr unsigned GELU_APPX_REPLAY_START = 0;
inline constexpr unsigned GELU_APPX_REPLAY_LEN = 4;

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_gelu_appx() {
    static_assert(ITERATIONS >= 1, "calculate_gelu_appx needs at least one pass");

    sfpi::l_reg[GELU_APPX_HALF_LREG] = sfpi::vFloat(0.5f);
    {
        sfpi::vFloat half = sfpi::l_reg[GELU_APPX_HALF_LREG];
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::l_reg[GELU_APPX_RESULT_LREG] =
            in * half + sfpi::lut<sfpi::LutMode::Fp16x6_HWM3>(in, sfpi::LutSign::Update);
        sfpi::l_reg[GELU_APPX_HALF_LREG] = half;
    }
    // Body for pass r: load pass r+1, store pass r (Dest += SFP_ROWS), lut pass r+1, mad pass r+1.
    TTI_REPLAY(GELU_APPX_REPLAY_START, GELU_APPX_REPLAY_LEN, 0, 0, 0, 1);
    {
        sfpi::vFloat half = sfpi::l_reg[GELU_APPX_HALF_LREG];
        sfpi::vFloat result = sfpi::l_reg[GELU_APPX_RESULT_LREG];
        sfpi::vFloat in = sfpi::dst_reg[1];
        sfpi::dst_reg[0].mode(ADDR_MOD_6) = result;
        sfpi::l_reg[GELU_APPX_RESULT_LREG] =
            in * half + sfpi::lut<sfpi::LutMode::Fp16x6_HWM3>(in, sfpi::LutSign::Update);
        sfpi::l_reg[GELU_APPX_HALF_LREG] = half;
    }
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        TTI_REPLAY(GELU_APPX_REPLAY_START, GELU_APPX_REPLAY_LEN, 0, 0, 0, 0);
    }
    sfpi::dst_reg[0].mode(ADDR_MOD_6) = sfpi::vFloat(sfpi::l_reg[GELU_APPX_RESULT_LREG]);
}

// =============================================================================
// Forward GELU - Piecewise CDF Approximation
// =============================================================================
// GELU(x) = x * Phi(x) where Phi(x) = 0.5*(1+erf(x/sqrt(2))) is the CDF
//
// Three active regions (plus zero default):
//   x >= 2.78125:                  Identity (result = x)
//   -3.125 <= x < 2.78125:         Core CDF polynomial (degree-13 in u=x²)
//   -5.54259443 < x < -3.125:      Moroz exp_21f: GELU = x · round_to_grid(H), H = 2^(Q(x) - 127)
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

// Every constant is reloaded each pass: one SFPLOADI if it is exact in bf16 or fp16a, two otherwise.
// So all coefficients are fitted on those grids, except the four marked Prgm, which gelu_init puts in
// the programmable constant registers. Fitted on all bf16 inputs for the fewest results that differ
// from torch: 37 of 58298 normal inputs, each by 1 ULP.

// Degree-13 CDF polynomial for Phi(x) over [-3.125, 2.78125]
// Phi(x) = 0.5 + x * p(x²); only odd powers.
constexpr float GELU_CDF_CORE_C0 = 5.000000000e-01f;
constexpr float GELU_CDF_CORE_C1 = 3.9887374640e-01f;    // Prgm0
constexpr float GELU_CDF_CORE_C3 = -6.6067621112e-02f;   // Prgm1
constexpr float GELU_CDF_CORE_C5 = 9.5703462139e-03f;    // Prgm2
constexpr float GELU_CDF_CORE_C7 = -1.0137557983e-03f;   // fp16a
constexpr float GELU_CDF_CORE_C9 = 7.3492527008e-05f;    // fp16a
constexpr float GELU_CDF_CORE_C11 = -3.2186508179e-06f;  // bf16
constexpr float GELU_CDF_CORE_C13 = 6.3329935074e-08f;   // bf16

// Exp region: H(x) = ½·erfc(|x|/√2) = 2^(Q(x) - 127), Q quadratic in x (log2 error 1.7e-3).
constexpr float GELU_EXP_Q0 = 1.2540007019e+02f;   // Prgm3
constexpr float GELU_EXP_Q1 = 5.7226562500e-01f;   // bf16
constexpr float GELU_EXP_Q2 = -6.9091796875e-01f;  // fp16a
// 2^frac ≈ 1 + C1·m + C2·m², m the 23-bit mantissa as an integer.
constexpr float GELU_EXP_C1 = 7.8231096268e-08f;  // bf16
constexpr float GELU_EXP_C2 = 4.7739590059e-15f;  // bf16

// Same bf16 inputs as -5.54259443 (the first x where libm erff = -1), but fp16a-exact.
constexpr float GELU_SAT_BF16 = -5.546875f;

// Forward GELU Evaluation with CDF Polynomial Approximation
// GELU(x) = x * Phi(x) where Phi is approximated piecewise
sfpi_inline sfpi::vFloat calculate_gelu_piecewise(sfpi::vFloat x) {
    sfpi::vFloat result = 0.0f;  // Default: 0 for x <= -5.54259443 (torch saturation)
    sfpi::vFloat x2 = x * x;

    v_if(x > GELU_SAT_BF16) {
        // Core polynomial first, as an interleaved even/odd split in x⁴, so its MADs overlap the exp
        // chain below. PolynomialEvaluator's split of 7 coefficients would add a C11 + x⁴·0 MAD.
        sfpi::vFloat x4 = x2 * x2;
        sfpi::vFloat even = GELU_CDF_CORE_C9 + x4 * GELU_CDF_CORE_C13;
        sfpi::vFloat odd = GELU_CDF_CORE_C7 + x4 * GELU_CDF_CORE_C11;
        even = sfpi::vConstFloatPrgm2 + x4 * even;
        odd = sfpi::vConstFloatPrgm1 + x4 * odd;
        even = sfpi::vConstFloatPrgm0 + x4 * even;
        sfpi::vFloat odd_poly = even + x2 * odd;
        sfpi::vFloat phi = GELU_CDF_CORE_C0 + x * odd_poly;

        // H = 2^(Q(x) - 127) via Moroz exp_21f.
        sfpi::vFloat xlog2 = PolynomialEvaluator::eval(x, sfpi::vConstFloatPrgm3, GELU_EXP_Q1, GELU_EXP_Q2);

        sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);
        sfpi::vInt exponential_part = sfpi::exexp(sfpi::as<sfpi::vFloat>(z), sfpi::ExponentMode::Biased);
        sfpi::vMag fractional_part = sfpi::exman(sfpi::as<sfpi::vFloat>(z));

        sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
        frac = PolynomialEvaluator::eval(frac, 1.0f, GELU_EXP_C1, GELU_EXP_C2);
        sfpi::vFloat H = sfpi::setexp(frac, exponential_part);

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
        // calculate_gelu_appx stores through ADDR_MOD_6: Dest += SFP_ROWS per pass.
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ckernel::math::SFP_ROWS}}.set(ADDR_MOD_6);

        // Segment slopes/intercepts, packed hi/lo the way SFPLUTFP32's FP16 6-entry mode 1 reads
        // them. Same values as the Blackhole table (l_reg0 = 0x37E7322B, l_reg4 = 0xB12286D8, ...).
        sfpi::lut_init(
            sfpi::sLut16si(0.1928f, -0.00010443f),
            sfpi::sLut16si(0.4939f, -0.1604f),
            sfpi::sLut16si(0.6188f, -0.2795f),
            sfpi::sLut16si(0.6099f, -0.2635f),
            sfpi::sLut16si(0.5402f, -0.1194f),
            sfpi::sLut16si(0.5000f, 0.0f));
    } else if constexpr (is_fp32_dest_acc_en) {
        // FP32 accurate mode: rational erf evaluation requires reciprocal init
        _init_reciprocal_<false>();
    } else {
        // BF16 accurate mode: the full-fp32 coefficients of calculate_gelu_piecewise.
        sfpi::vConstFloatPrgm0 = GELU_CDF_CORE_C1;
        sfpi::vConstFloatPrgm1 = GELU_CDF_CORE_C3;
        sfpi::vConstFloatPrgm2 = GELU_CDF_CORE_C5;
        sfpi::vConstFloatPrgm3 = GELU_EXP_Q0;
    }
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

// Whether BF16 DEST runs the generated gelu kernel as one call over the whole tile.
inline constexpr bool gelu_bf16_whole_tile = false;

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

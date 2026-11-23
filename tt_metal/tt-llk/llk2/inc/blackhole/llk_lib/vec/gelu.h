// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, vec: gelu and gelu_tanh (llk_lib/vec/common.h has the shape of a function file).
// gelu: x * Phi(x) through a rational erf (32-bit Dest), a piecewise form (16-bit Dest) or a lookup (fast).
// gelu_tanh: the tanh form, exact (llk_lib/vec/tanh.h) or fast (an exp2 and two Newton steps).
// Moved from tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h and ckernel_sfpu_piecewise_rational.h.

#pragma once

#include <cstdint>

#include "llk_lib/vec/common.h"
#include "llk_lib/vec/exp.h"
#include "llk_lib/vec/recip.h"
#include "llk_lib/vec/tanh.h"

#ifdef LLK2_SFPU_THREAD

namespace llk2::llk_lib
{

// ============================================================================
// Parity x²-Horner: odd num / even den → evaluate in x² basis
// Halves the multiply-add count. In LLK 2.0 the fp32 gelu's erf uses it.
// ============================================================================

template <std::uint32_t NUM_DEGREE, std::uint32_t DEN_DEGREE>
sfpi_inline void vec_rational_parity_numer_denom(
    const float* num_coeffs, const float* den_coeffs, sfpi::vFloat x, sfpi::vFloat x2, sfpi::vFloat& out_numer, sfpi::vFloat& out_denom)
{
    // NUM_TOP/DEN_TOP: highest odd/even index used in x²-Horner.
    // If NUM_DEGREE is even, the leading (even-index) coeff must be zero — we skip it.
    constexpr int NUM_TOP   = (NUM_DEGREE % 2 == 1) ? NUM_DEGREE : NUM_DEGREE - 1;
    constexpr int DEN_TOP   = (DEN_DEGREE % 2 == 0) ? DEN_DEGREE : DEN_DEGREE - 1;
    constexpr int NUM_STEPS = (NUM_TOP - 1) / 2;
    constexpr int DEN_STEPS = DEN_TOP / 2;

    sfpi::vFloat numerator = num_coeffs[NUM_TOP];
    sfpi::vFloat denom     = den_coeffs[DEN_TOP];

    if constexpr (NUM_STEPS > DEN_STEPS)
    {
#pragma GCC unroll 64
        for (int k = 0; k < NUM_STEPS - DEN_STEPS; k++)
        {
            numerator = numerator * x2 + num_coeffs[NUM_TOP - 2 * (k + 1)];
        }
    }
    else if constexpr (DEN_STEPS > NUM_STEPS)
    {
#pragma GCC unroll 64
        for (int k = 0; k < DEN_STEPS - NUM_STEPS; k++)
        {
            denom = denom * x2 + den_coeffs[DEN_TOP - 2 * (k + 1)];
        }
    }

    constexpr int MIN_STEPS = (NUM_STEPS < DEN_STEPS) ? NUM_STEPS : DEN_STEPS;
    constexpr int NUM_POS   = NUM_TOP - 2 * ((NUM_STEPS > DEN_STEPS) ? (NUM_STEPS - DEN_STEPS) : 0);
    constexpr int DEN_POS   = DEN_TOP - 2 * ((DEN_STEPS > NUM_STEPS) ? (DEN_STEPS - NUM_STEPS) : 0);

#pragma GCC unroll 64
    for (int k = 1; k <= MIN_STEPS; k++)
    {
        numerator = numerator * x2 + num_coeffs[NUM_POS - 2 * k];
        denom     = denom * x2 + den_coeffs[DEN_POS - 2 * k];
    }

    out_numer = numerator * x; // odd parity: P(x) = x * Horner_result
    out_denom = denom;
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
// Result: ULP ≤ 1 across the tail (0 ULP for almost all of it).
//
// v_if/v_and narrowing: widest region first, each v_and overwrites for narrower set.
// =============================================================================

// Degree-13 CDF polynomial for Phi(x) over [-3.125, 2.78125]
// Phi(x) = 0.5 + x * p(x²); only odd powers, evaluated via u=x² Horner (7 MADs).
// Minimax-fitted with BF16-ULP weighting → MaxULP = 0.87 < 1.
constexpr float vec_gelu_cdf_core_c0  = 5.000000000e-01f;
constexpr float vec_gelu_cdf_core_c1  = 3.9894227818e-01f;
constexpr float vec_gelu_cdf_core_c3  = -6.6361041488e-02f;
constexpr float vec_gelu_cdf_core_c5  = 9.7720050615e-03f;
constexpr float vec_gelu_cdf_core_c7  = -1.0717806322e-03f;
constexpr float vec_gelu_cdf_core_c9  = 8.1812159812e-05f;
constexpr float vec_gelu_cdf_core_c11 = -3.8082057209e-06f;
constexpr float vec_gelu_cdf_core_c13 = 7.9821413868e-08f;

// Degree-3 H-form correction polynomial for the exp-based region (-5.54259443, -3.125].
// H(x) = ½·erfc(|x|/√2) ≈ exp(-x²/2) · corr_H(x), and GELU(x) = x · H(x).
// Fitted (uniform relative error) so the deep-tail snap is exact and the smooth
// region stays ULP ≤ 1. See deep-tail snap below for why the H-form matters.
constexpr float vec_gelu_hcorr_c0 = 3.0369991064e-01f;
constexpr float vec_gelu_hcorr_c1 = 9.5413386822e-02f;
constexpr float vec_gelu_hcorr_c2 = 1.3809983619e-02f;
constexpr float vec_gelu_hcorr_c3 = 7.5950479368e-04f;

// -0.5 / ln(2): folds the -x²/2 of the exp region into the exp_21f scale.
constexpr float vec_gelu_neg_half_one_ln2 = -0.72134752044f;

// Forward GELU Evaluation with CDF Polynomial Approximation
// GELU(x) = x * Phi(x) where Phi is approximated piecewise
//
// The loop-invariant constants the caller hoists are passed in (hoisted constants: llk_lib/vec/common.h): exp_21f's c0 and c1 are
// read from vConstFloatPrgm1/2 (programmed by vec_gelu_init<false, false>); -0.5/ln2 and the two highest CDF
// coefficients are sfpi::vFloats the caller built once before its loop.
template <typename K, typename C13, typename C11>
sfpi_inline sfpi::vFloat vec_gelu_piecewise(sfpi::vFloat x, K neg_half_one_ln2, C13 cdf13, C11 cdf11)
{
    sfpi::vFloat result = 0.0f; // Default: 0 for x <= -5.54259443 (torch saturation)
    sfpi::vFloat x2     = x * x;

    v_if (x > -5.54259443f)
    {
        // Shared smooth H ≈ ½·erfc(|x|/√2) via Moroz exp_21f · corr_H.
        sfpi::vFloat xlog2 = x2 * neg_half_one_ln2 + 127.0f;

        sfpi::vInt z                = vec_exp_21f_to_int(xlog2);
        sfpi::vInt exponential_part = sfpi::exexp(sfpi::as<sfpi::vFloat>(z), sfpi::ExponentMode::Biased);
        sfpi::vMag fractional_part  = sfpi::exman(sfpi::as<sfpi::vFloat>(z));

        sfpi::vFloat frac    = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
        frac                 = vec_polynomial::eval(frac, sfpi::vConstFloatPrgm1, sfpi::vConstFloatPrgm2, vec_exp_21f_c2);
        sfpi::vFloat exp_val = sfpi::setexp(frac, exponential_part);

        sfpi::vFloat H = exp_val * vec_polynomial::eval(x, vec_gelu_hcorr_c0, vec_gelu_hcorr_c1, vec_gelu_hcorr_c2, vec_gelu_hcorr_c3);

        // Exp region (-5.54259443, -3.125]: GELU = x · round_to_grid(H).
        // Round H to the nearest multiple of 2⁻²⁵: adding 0.375 = (2²³+2²²)·2⁻²⁵ shifts H's
        // 2⁻²⁵ place to the FP32 round-to-nearest-even boundary (safe since H ∈ [3e-8, 9e-4] ≪ 0.25),
        // so (H + 0.375) - 0.375 == round(H / 2⁻²⁵) · 2⁻²⁵. In the deep tail (-5.5426, -4.828]
        // this reproduces torch's exact float32 ercc staircase (0 BF16 ULP); above -4.828 the
        // 2⁻²⁵ grid is far finer than BF16 so rounding is lossless. One constant serves the
        // whole exp region — no separate branch or multiply.
        constexpr float ROUND_TO_GRID = 0.375f;
        sfpi::vFloat Hs               = (H + ROUND_TO_GRID) - ROUND_TO_GRID;
        result                        = x * Hs;

        // Core CDF region [-3.125, 2.78125): GELU = x · Phi_core(x).
        v_and(x >= -3.125f);
        sfpi::vFloat odd_poly = vec_polynomial::eval(
            x2, vec_gelu_cdf_core_c1, vec_gelu_cdf_core_c3, vec_gelu_cdf_core_c5, vec_gelu_cdf_core_c7, vec_gelu_cdf_core_c9, cdf11, cdf13);
        sfpi::vFloat phi = vec_gelu_cdf_core_c0 + x * odd_poly;
        result           = x * phi;

        // Identity region x >= 2.78125.
        v_and(x >= 2.78125f);
        result = x;
    }
    v_endif;

    return result;
}

template <bool Fast, bool Fp32>
void vec_gelu_init()
{
    if constexpr (Fast)
    {
        sfpi::vConstFloatPrgm0 = 0.5f;

        // Load the packed 6-segment LUT coefficients directly into the LRegs via sfpi.
        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut16ss(0.1928f, 0.4939f);
        sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vLut16ii(-0.00010443f, -0.1604f);

        sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut16ss(0.6188f, 0.6099f);
        sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vLut16ii(-0.2795f, -0.2635f);

        sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut16ss(0.5402f, 0.5f);
        sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vLut16ii(-0.1194f, 0.0f);
    }
    else if constexpr (Fp32)
    {
        // FP32 accurate mode: rational erf evaluation requires reciprocal init
        vec_recip_newton_init<false>();
    }
    else
    {
        // BF16 accurate mode: no reciprocal; Prgm1/2 hold exp_21f's c0 and c1 for vec_gelu_piecewise
        // (its other loop constants live in LRegs, see vec_gelu_iterations).
        sfpi::vConstFloatPrgm1 = vec_exp_21f_c0;
        sfpi::vConstFloatPrgm2 = vec_exp_21f_c1;
    }
}

template <int Iterations>
inline void vec_gelu_lut_iterations()
{
    sfpi::vLut16ss s01 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut16ss s23 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut16ss s45 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vLut16ii i01 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vLut16ii i23 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vLut16ii i45 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < Iterations; d++)
    {
        sfpi::vFloat in      = sfpi::dst_reg[0];
        sfpi::vFloat half    = sfpi::vConstFloatPrgm0;
        sfpi::vFloat half_in = in * half;
        sfpi::vFloat result  = sfpi::lut(in, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Update);
        result               = half_in + result;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = s01;
    sfpi::l_reg[sfpi::LRegs::LReg1] = s23;
    sfpi::l_reg[sfpi::LRegs::LReg2] = s45;
    sfpi::l_reg[sfpi::LRegs::LReg4] = i01;
    sfpi::l_reg[sfpi::LRegs::LReg5] = i23;
    sfpi::l_reg[sfpi::LRegs::LReg6] = i45;
}

// FP32 erf: n16/d16 parity rational coefficients, MaxULP=1 vs FP64.
// Split from ERF_LUT (ckernel_sfpu_erf.h INP_FLOAT32 branch) entries [2..18] (num) and [19..35] (den).
// vec_rational_parity_numer_denom() takes const float* — no std::array needed.
constexpr float vec_gelu_erf_num[17] = { // odd powers only (c0=0, c2=0, ..., c16=0)
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

constexpr float vec_gelu_erf_den[17] = { // even powers only (c1=0, c3=0, ..., c15=0)
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

template <bool Fast, bool Fp32, int Iterations = 8>
inline void vec_gelu_iterations()
{
    if constexpr (Fast)
    {
        vec_gelu_lut_iterations<Iterations>();
    }
    else if constexpr (Fp32)
    {
        // FP32 accurate mode: GELU(x) = x * 0.5 * (1 + erf(x/√2))
        // using the n16/d16 piecewise rational erf (MaxULP=1 vs FP64).
        constexpr float INV_SQRT2    = 0.7071067811865475f;
        constexpr float vec_gelu_sat = -5.54259443f; // 0xc0b15cef: first x where libm erff=-1.0
#pragma GCC unroll 0
        for (int d = 0; d < Iterations; d++)
        {
            sfpi::vFloat x = sfpi::dst_reg[0];
            // Default for lanes that fail `x > vec_gelu_sat` below. x - x is 0 for finite x and NaN for
            // NaN, so a negative NaN (which also fails the compare) still propagates.
            sfpi::vFloat result = x - x;
            v_if (x > vec_gelu_sat)
            {
                sfpi::vFloat scaled = x * INV_SQRT2;
                scaled              = sfpi::min(scaled, 10.0f);
                sfpi::vFloat x2     = scaled * scaled;
                sfpi::vFloat erf_n, erf_d;
                vec_rational_parity_numer_denom<16, 16>(vec_gelu_erf_num, vec_gelu_erf_den, scaled, x2, erf_n, erf_d);
                sfpi::vFloat erf_val = erf_n * vec_recip_value<false>(erf_d);
                erf_val              = sfpi::clamp(erf_val, -1.0f, 1.0f);
                result               = x * (0.5f + 0.5f * erf_val);
                // Stuck-erff guard: when rational rounds erf to -1.0 inside computation zone,
                // glibc/torch overestimates erfc → first stuck level = x * 2^-25.
                constexpr float HALF_ULP_AT_1 = 2.9802322387695313e-08f; // 2^-25
                v_if (erf_val == -1.0f)
                {
                    result = x * HALF_ULP_AT_1;
                }
                v_endif;
            }
            v_endif;
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
    else
    {
        // BF16 accurate mode: piecewise CDF with Max ULP=1 vs true GELU.
        // unroll 8 fills the SFPU pipeline across 8 independent dst-tile chains.
        // -0.5/ln2 and the two highest CDF coefficients are loaded once and kept in LRegs for the whole
        // loop; the exp_21f coefficients come from Prgm1/2 (vec_gelu_init). Prgm0 is left alone: the LReg form
        // is one instruction shorter here and keeps the reciprocal's convention for that register.
        sfpi::vFloat neg_half_one_ln2 = vec_gelu_neg_half_one_ln2;
        sfpi::vFloat cdf13 = vec_gelu_cdf_core_c13, cdf11 = vec_gelu_cdf_core_c11;
#pragma GCC unroll 8
        for (int d = 0; d < Iterations; d++)
        {
            sfpi::vFloat in     = sfpi::dst_reg[0];
            sfpi::vFloat result = vec_gelu_piecewise(in, neg_half_one_ln2, cdf13, cdf11);
            result              = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[0]    = result;
            sfpi::dst_reg++;
        }
    }
}

// =============================================================================
// GELU tanh approximation in FP32:
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// =============================================================================

template <bool Fp32, int Iterations = 8>
inline void vec_gelu_tanh_exact_iterations()
{
    constexpr float SQRT_2_OVER_PI  = 0.7978845608028654f;
    constexpr float vec_gelu_tanh_k = 0.044715f;

#pragma GCC unroll 8
    for (int d = 0; d < Iterations; d++)
    {
        sfpi::vFloat x = sfpi::dst_reg[0];

        sfpi::vFloat x2 = x * x;
        sfpi::vFloat p  = vec_gelu_tanh_k * x2 + 1.0f;
        sfpi::vFloat q  = x * p;
        sfpi::vFloat u  = SQRT_2_OVER_PI * q;
        sfpi::vFloat t  = vec_tanh_fp32(u);

        // reload due to register pressure
        x                   = sfpi::dst_reg[0];
        sfpi::vFloat half_x = 0.5f * x;
        sfpi::vFloat result = half_x * t + half_x;

        if constexpr (!Fp32)
        {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

inline void vec_gelu_tanh_exact_init()
{
    // initialise constants for vec_tanh_fp32
    vec_tanh_init<false, true>();
}

// =============================================================================
// Fast GELU tanh approximation (opt-in, BF16-grade):
//   0.5 * x * (1 + tanh(u)) == x * sigmoid(2u) == x / (1 + exp(-2u)),  u = sqrt(2/pi) * (x + 0.044715 * x^3)
// exp(-2u) = 2^(-2u * log2(e)) uses the exp_21f bit trick (vec_exp_21f_to_int) with a degree-4
// polynomial for 2^f (max relative error 1.5e-5, below; exp_21f's quadratic is 1.7e-3), then two Newton steps on
// the reciprocal. Max error vs torch gelu(approximate="tanh") is ~1e-6 absolute, <= 1 BF16 ULP, in about
// 22 SFPU ops against ~38 for vec_gelu_tanh_exact_iterations. For FP32 outputs, prefer vec_gelu_tanh_exact_iterations.
// =============================================================================

// 2^f on [0, 1) as 1 + c1 f + ... + c4 f^4, fitted for relative error (max 1.5e-5) with zero mean error: a one-sided
// error biases |GELU| in one direction, which accumulates through a residual stream. c0 = 1 and c2..c4 are rounded
// to BF16 so each loads with one SFPLOADI; c1 lives in a programmable constant. The 2^-23 mantissa scale is folded in
// (c_k * 2^(-23k), which keeps them BF16-exact). setexp() below overwrites the exponent, so the polynomial must stay
// in [1, 2) for every mantissa: it spans [1, 1.99997] over all 2^23.
constexpr float vec_gelu_tanh_fast_exp2_c1 = 8.2622300113e-08f; // vConstFloatPrgm2
constexpr float vec_gelu_tanh_fast_exp2_c2 = 3.4278135885e-15f;
constexpr float vec_gelu_tanh_fast_exp2_c3 = 8.8508325543e-23f;
constexpr float vec_gelu_tanh_fast_exp2_c4 = 2.7117093617e-30f;

// log2 of exp(-2u) = x * (A + B * x^2), u = sqrt(2/pi) * (x + 0.044715 * x^3).
constexpr float vec_gelu_tanh_fast_a = -2.3022081852e+00f; // -2 * sqrt(2/pi) * log2(e), vConstFloatPrgm0
constexpr float vec_gelu_tanh_fast_b = -1.0294324160e-01f; // A * 0.044715, vConstFloatPrgm1

template <bool Fp32, int Iterations = 8>
inline void vec_gelu_tanh_fast_iterations()
{
#pragma GCC unroll 8
    for (int d = 0; d < Iterations; d++)
    {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Biased log2 of exp(-2u). Clamped so the exponent stays finite: 0 gives exp(-2u) ~ 0 (result x),
        // 254 gives exp(-2u) ~ 2^127, whose reciprocal flushes to 0 (result 0).
        sfpi::vFloat x2    = x * x;
        sfpi::vFloat t     = x2 * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0;
        sfpi::vFloat xlog2 = x * t + 127.0f;
        xlog2              = sfpi::clamp(xlog2, 0.0f, 254.0f);

        sfpi::vFloat z              = sfpi::as<sfpi::vFloat>(vec_exp_21f_to_int(xlog2));
        sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
        sfpi::vMag fractional_part  = sfpi::exman(z);
        sfpi::vFloat frac           = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
        frac = vec_polynomial::eval(frac, 1.0f, sfpi::vConstFloatPrgm2, vec_gelu_tanh_fast_exp2_c2, vec_gelu_tanh_fast_exp2_c3, vec_gelu_tanh_fast_exp2_c4);
        sfpi::vFloat e = sfpi::setexp(frac, exponential_part);

        // d >= 1, so no zero/inf/NaN cases: approximate reciprocal plus two branch-free Newton steps. Each step
        // lands at or below 1/d, so a single step would bias |GELU| low; the second makes that bias negligible.
        sfpi::vFloat den    = e + 1.0f;
        sfpi::vFloat r      = sfpi::approx_recip(den);
        r                   = r * (2.0f - den * r);
        r                   = r * (2.0f - den * r);
        sfpi::vFloat result = x * r;

        if constexpr (!Fp32)
        {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

inline void vec_gelu_tanh_fast_init()
{
    sfpi::vConstFloatPrgm0 = vec_gelu_tanh_fast_a;
    sfpi::vConstFloatPrgm1 = vec_gelu_tanh_fast_b;
    sfpi::vConstFloatPrgm2 = vec_gelu_tanh_fast_exp2_c1;
}

/// gelu's tanh form, its part of vec_configure: the fast or the exact one's. Fp32 is not read.
template <bool Fast, bool Fp32>
sfpi_inline void vec_gelu_tanh_init()
{
    if constexpr (Fast)
    {
        vec_gelu_tanh_fast_init();
    }
    else
    {
        vec_gelu_tanh_exact_init();
    }
}

/// gelu's tanh form on Iterations iterations: the fast (bf16-grade) or the exact one.
template <bool Fast, bool Fp32, int Iterations = 8>
sfpi_inline void vec_gelu_tanh_iterations()
{
    if constexpr (Fast)
    {
        vec_gelu_tanh_fast_iterations<Fp32, Iterations>();
    }
    else
    {
        vec_gelu_tanh_exact_iterations<Fp32, Iterations>();
    }
}

} // namespace llk2::llk_lib

#endif // LLK2_SFPU_THREAD

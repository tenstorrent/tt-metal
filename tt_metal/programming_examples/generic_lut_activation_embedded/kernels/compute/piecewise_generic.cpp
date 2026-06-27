// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

// Canonical eval_method taxonomy. Translates the single EVAL_METHOD_* selector
// (+ EXPONENT_ALU_* / REDUCE_* sub-tags) emitted by codegen into the legacy
// feature macros the kernel bodies below were written against. Must precede any
// RANGE_REDUCTION_* / EVAL_METHOD_* read.
#include "eval_method.h"

// Include reciprocal function for tan range reduction (tan_expand needs -1/poly),
// the exp2 sigmoid compose (1/(1+exp(-x))), sigmoid-product compose
// (x/(1+exp(-x))), and the pow path's final 1/result
// (rsqrt = 1/sqrt(x), tagged expalu_reciprocal -> POW_HW_RECIPROCAL).
#if (                                                                                                             \
    defined(RANGE_REDUCTION_TAN) || defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || \
    defined(POW_HW_RECIPROCAL)) &&                                                                                \
    defined(TRISC_MATH)
#include "ckernel_sfpu_recip.h"
#endif

namespace kutil = norm::kernel_util;

/**
 * Generic Piecewise Polynomial LUT Activation
 *
 * This kernel supports ANY polynomial degree (0-16) and ANY depth (4-64 segments).
 * Each segment is approximated as: y = c0 + c1*x + c2*x² + ... + cn*x^n
 * Evaluated using Horner's scheme for numerical stability.
 *
 * Template parameters (compile-time):
 *   POLY_DEGREE   - Polynomial degree (0=constant, 1=linear, 2=quadratic, ... 16=hexadecic)
 *   NUM_SEGMENTS  - Number of segments (4, 8, 16, 32, 64, etc.)
 *   LUT_SIZE      - Total LUT size = (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)
 *
 * LUT Format: [b0, b1, ..., bN, c0_seg0, c1_seg0, ..., cn_seg0, c0_seg1, ...]
 * where bi are boundary points and cj are polynomial coefficients
 *
 * Benefits:
 *   - Single implementation for all degree/depth combinations
 *   - Zero runtime overhead (all branching at compile-time)
 *   - Automatic support for new configurations
 *   - Reduced code duplication (7 files → 1 file)
 *
 * Supports both modes:
 *   - EMBEDDED_LUT: Coefficients compiled directly into kernel (zero L1 overhead)
 *   - CB mode: Coefficients loaded from L1 circular buffer (runtime flexibility)
 */

#ifdef TRISC_MATH

// Range reduction: Cody-Waite method for exp/trig, mantissa/exponent extraction for log/cbrt
// Only included when a reduction/standalone eval_method or ASYMPTOTIC_FACTOR_* is active.
#if EVAL_METHOD_NEEDS_REDUCTION_HELPERS || defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || \
    defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR)
#include "sfpu/ckernel_sfpu_converter.h"
#endif

namespace sfpi {

inline vFloat apply_output_postcompose(vFloat y) {
#if defined(POSTCOMPOSE_AFFINE_Y)
    y = POSTCOMPOSE_B * y + POSTCOMPOSE_A;
#endif
    return y;
}

#ifdef RANGE_REDUCTION_EXP
// Cody-Waite range reduction: x → (s, k_int) where exp(x) = 2^k * exp(s)
// s ∈ [-ln(2)/2, ln(2)/2] ≈ [-0.347, 0.347]
inline void exp_reduce(vFloat x, vFloat& s, vInt& k_int) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    vFloat z = x * INV_LN2;

    // Round z to nearest integer (branch-free magic-number technique from ckernel_sfpu_exp.h)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat k = tmp - c231;
    k_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);

    // Cody-Waite extended precision: s = x - k*ln2
    // Using negated constants so compiler can emit single SFPMAD instructions
    constexpr float NEG_LN2_HI = -0.6931152343750000f;
    constexpr float NEG_LN2_LO = -3.19461832987e-05f;
    vFloat s_hi = k * NEG_LN2_HI + x;
    s = k * NEG_LN2_LO + s_hi;
}

// Reconstruct: result = poly_result * 2^k via exponent bit manipulation
inline vFloat exp_expand(vFloat poly_result, vInt k_int) {
    vInt p_exp = exexp_nodebias(poly_result);
    vInt new_exp = p_exp + k_int;
    return setexp(poly_result, new_exp);
}
#endif

#ifdef RANGE_REDUCTION_TRIG
// Trig range reduction: x → (s, q_int) where f(x) = (-1)^q * f(s)
// s ∈ [-π/2, π/2], works for both sin and cos
inline void trig_reduce(vFloat x, vFloat& s, vInt& q_int) {
    constexpr float FRAC_1_PI = 0.31830988618379067f;
    vFloat z = x * FRAC_1_PI;

    // Round z to nearest integer (branch-free magic-number technique)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat q = tmp - c231;
    q_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);

    // Cody-Waite: s = x - q*π with extended precision
    constexpr float NEG_PI_HI = -3.140625f;
    constexpr float NEG_PI_LO = -0.00096765358979323846f;
    vFloat s_hi = q * NEG_PI_HI + x;
    s = q * NEG_PI_LO + s_hi;
}

// Reconstruct: sin(x + kπ) = (-1)^k * sin(x), cos(x + kπ) = (-1)^k * cos(x)
inline vFloat trig_expand(vFloat poly_result, vInt q_int) {
    v_if(q_int & 1) { poly_result = -poly_result; }
    v_endif;
    return poly_result;
}
#endif

#if defined(EVAL_METHOD_TRIG_RESIDUAL)
template <uint32_t DEGREE>
inline vFloat trig_residual_odd_eval(vFloat x) {
#if defined(TRIG_RESIDUAL_PHASE_COSINE_PI2_ODD)
    vFloat half = sFloat16b(0.5f);
    vFloat inv_pi = vConstFloatPrgm2;
    vFloat z = __builtin_rvtt_sfpmad(x.get(), inv_pi.get(), half.get(), SFPMAD_MOD1_OFFSET_NONE);

    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat q = tmp - c231;
    vInt q_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);

    vFloat two = sFloat16b(2.0f);
    vFloat neg_one = vConstNeg1;
    vFloat j = __builtin_rvtt_sfpmad(q.get(), two.get(), neg_one.get(), SFPMAD_MOD1_OFFSET_NONE);
    constexpr float NEG_PI_2_P0 = -0x1.92p+0f;
    constexpr float NEG_PI_2_P1 = -0x1.fbp-12f;
    vFloat a = j * NEG_PI_2_P0 + x;
    a = j * NEG_PI_2_P1 + a;
#elif defined(TRIG_RESIDUAL_PHASE_SINE_PI_ODD)
    vFloat inv_pi = vConstFloatPrgm2;
    vFloat z = x * inv_pi;

    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat j = tmp - c231;
    vInt q_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);

    constexpr float NEG_PI_P0 = -0x1.92p+1f;
    constexpr float NEG_PI_P1 = -0x1.fbp-11f;
    vFloat a = j * NEG_PI_P0 + x;
    a = j * NEG_PI_P1 + a;
#else
#error "EVAL_METHOD_TRIG_RESIDUAL requires a supported TRIG_RESIDUAL_PHASE_*"
#endif
    a = j * vConstFloatPrgm0 + a;
    a = j * vConstFloatPrgm1 + a;

    q_int <<= 31;
    a = reinterpret<vFloat>(reinterpret<vInt>(a) ^ q_int);

    vFloat s = a * a;
    if constexpr (DEGREE == 7) {
        vFloat r = TRIG_RESIDUAL_COEFFS[7] * s + TRIG_RESIDUAL_COEFFS[5];
        vFloat c = a * s;
        r = r * s + TRIG_RESIDUAL_COEFFS[3];
#ifdef TRIG_RESIDUAL_C1_IS_ONE
        return r * c + a;
#else
        return r * c + TRIG_RESIDUAL_COEFFS[1] * a;
#endif
    } else if constexpr (DEGREE == 5) {
        vFloat r = TRIG_RESIDUAL_COEFFS[5] * s + TRIG_RESIDUAL_COEFFS[3];
        vFloat c = a * s;
#ifdef TRIG_RESIDUAL_C1_IS_ONE
        return r * c + a;
#else
        return r * c + TRIG_RESIDUAL_COEFFS[1] * a;
#endif
    } else {
        constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
        constexpr int STEPS = (TOP - 1) / 2;
        vFloat r = TRIG_RESIDUAL_COEFFS[TOP];
#pragma GCC unroll 8
        for (int k = 1; k <= STEPS; k++) {
            r = r * s + TRIG_RESIDUAL_COEFFS[TOP - 2 * k];
        }
        return r * a;
    }
}
#endif

#ifdef RANGE_REDUCTION_TAN
// Tan range reduction: x → (a, j_int) where a ∈ [-π/4, π/4]
// Uses Cody-Waite reduction matching TTNN's sfpu_tan implementation
// j = round(x / (π/2)), a = x - j * (π/2)
inline void tan_reduce(vFloat x, vFloat& a, vInt& j_int) {
    constexpr float FRAC_2_PI = 0.6366197723675814f;  // 2/π
    vFloat z = x * FRAC_2_PI;

    // Round z to nearest integer (branch-free magic-number technique)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat j = tmp - c231;
    j_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);

    // Cody-Waite: a = x - j*(π/2) with extended precision
    // Split π/2 into high and low parts for accuracy
    constexpr float NEG_PI_2_HI = -1.5703125f;              // -π/2 high bits (fits in bf16)
    constexpr float NEG_PI_2_LO = -0.0004837512969970703f;  // -π/2 remainder
    vFloat a_hi = j * NEG_PI_2_HI + x;
    a = j * NEG_PI_2_LO + a_hi;
}

// Reconstruct: tan(x) where x = a + j*(π/2)
// j even: tan(x) = tan(a) = poly(a)
// j odd:  tan(x) = -cot(a) = -1/tan(a) = -1/poly(a)
inline vFloat tan_expand(vFloat poly_result, vInt j_int) {
    v_if(j_int & 1) {
        // j is odd: result = -1/poly(a)
        // Use 3 Newton-Raphson iterations for better precision (vs 2 in sfpu_reciprocal<false>)
        poly_result = -ckernel::sfpu::sfpu_reciprocal_iter<3>(poly_result);
    }
    v_endif;
    return poly_result;
}
#endif

#if defined(EVAL_METHOD_TAN_STANDALONE)
template <uint32_t DEGREE>
inline vFloat tan_standalone_eval(vFloat x) {
    vFloat inv_pio2 = vConstFloatPrgm2;
    vFloat rounding_bias = sFloat16b(0x1.8p23f);
    vFloat j = __builtin_rvtt_sfpmad(x.get(), inv_pio2.get(), rounding_bias.get(), SFPMAD_MOD1_OFFSET_NONE);
    vInt i = reinterpret<vInt>(j);
    j += -rounding_bias;
    i <<= 31;

    constexpr float NEG_PI_2_P0 = -0x1.92p+0f;
    constexpr float NEG_PI_2_P1 = -0x1.fbp-12f;
    vFloat a = x + j * NEG_PI_2_P0;
    a = a + j * NEG_PI_2_P1;
    a = a + j * vConstFloatPrgm0;
    a = a + j * vConstFloatPrgm1;

    vFloat s = a * a;
    constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
    vFloat r = TAN_STANDALONE_COEFFS[TOP];
#pragma GCC unroll 8
    for (int k = 1; k <= (TOP - 1) / 2; k++) {
        r = r * s + TAN_STANDALONE_COEFFS[TOP - 2 * k];
    }
    r = r * a;

    v_if(i < 0) {
        vFloat t = approx_recip(r);
        vFloat e = -r * t + vConst1;
        r = -t * e - t;
    }
    v_endif;
    return r;
}
#endif

#if defined(RANGE_REDUCTION_LOG)
// Log range reduction: x → (m, e_int) where x = 2^e * m, m ∈ [1, 2)
// log(x) = e*ln(2) + log(m)
// Uses IEEE 754 bit manipulation: exponent extraction and mantissa normalization
inline void log_reduce(vFloat x, vFloat& m, vInt& e_int) {
    // Extract biased exponent: for x = 2^e * m, biased_exp = e + 127
    vInt biased_exp = exexp_nodebias(x);

    // Unbias: e = biased_exp - 127
    // But we want m in [1, 2), so we set exponent to 127 (bias)
    // This gives m = x / 2^e = x * 2^(-e) = x with exponent = 127
    e_int = biased_exp - 127;

    // Normalize mantissa to [1, 2) by setting exponent bits to 127
    m = setexp(x, 127);
}

// Reconstruct: log(x) = e*ln(2) + log(m)
// poly_result = log(m) where m ∈ [1, 2)
// Final result = e*ln(2) + poly_result
inline vFloat log_expand(vFloat poly_result, vInt e_int) {
    // LOG_EXPAND_CONSTANT is base-specific: ln(2) for log, 1.0 for log2, log10(2) for log10
#ifndef LOG_EXPAND_CONSTANT
#define LOG_EXPAND_CONSTANT 0.6931471805599453f
#endif
    constexpr float EXPAND_C = LOG_EXPAND_CONSTANT;
    // int32_to_float expects SIGN-MAGNITUDE format, not two's complement.
    // Negative exponents must be converted: twos complement → sign-magnitude.
    v_if(e_int < 0) { e_int = setsgn(~e_int + 1, 1); }
    v_endif;
    vFloat e_float = int32_to_float(e_int, RoundMode::Nearest);
    return e_float * EXPAND_C + poly_result;
}
#endif

#ifdef RANGE_REDUCTION_CBRT
// Cbrt range reduction: x → (m, q, r, sign) where cbrt(x) = sign * C[r] * cbrt(m) * 2^q
//
// For x = sign(x) * |x| where |x| = m * 2^e with m ∈ [1, 2):
//   e = 3*q + r  where r ∈ {0, 1, 2}
//   cbrt(|x|) = cbrt(m * 2^r) * 2^q = cbrt(m) * cbrt(2^r) * 2^q
//   cbrt(x) = sign(x) * C[r] * cbrt(m) * 2^q
//
// where C = [1.0, cbrt(2), cbrt(4)]
//
// Split into cbrt_reduce_m (trivial, 2 ops) + cbrt_reduce_qrs (heavier)
// to avoid SFPU register spills. cbrt_reduce's many inlined temporaries
// exceed the 8 LREG limit when combined with polynomial evaluation variables.

// Scale constants: C[r] = cbrt(2^r) for r = 0, 1, 2
constexpr float CBRT_SCALE_C0 = 1.0f;
constexpr float CBRT_SCALE_C1 = 1.2599210498948732f;  // cbrt(2)
constexpr float CBRT_SCALE_C2 = 1.5874010519681994f;  // cbrt(4)

// Phase 1: Extract mantissa m ∈ [1, 2) — trivial, no register pressure
inline vFloat cbrt_reduce_m(vFloat x) { return setexp(setsgn(x, 0), 127); }

// Reconstruct: cbrt(x) = sign * C[r] * poly(m) * 2^q
inline vFloat cbrt_expand(vFloat poly_result, vInt q, vInt r, vInt sign_bits) {
    // Look up scale factor C[r] = cbrt(2^r)
    vFloat scale = CBRT_SCALE_C0;  // default r=0
    v_if(r == 1) { scale = CBRT_SCALE_C1; }
    v_endif;
    v_if(r == 2) { scale = CBRT_SCALE_C2; }
    v_endif;

    // result = C[r] * poly(m)
    vFloat result = scale * poly_result;

    // Multiply by 2^q via exponent addition
    vInt result_exp = exexp_nodebias(result);
    result = setexp(result, result_exp + q);

    // Apply original sign (cbrt is an odd function)
    vInt result_bits = reinterpret<vInt>(result);
    result = reinterpret<vFloat>(result_bits | sign_bits);

    return result;
}
#endif

// ============================================================================
// Hardware-exponent-ALU range reduction (exp2 / log2 / pow)
//
// Instead of software Cody-Waite (float k-int + 2^k + Newton reciprocal), these
// paths decompose the input with the SFPU exponent ALU (exexp / exman / setexp /
// addexp) and evaluate a LOW-DEGREE polynomial on the reduced domain — the same
// algorithm TTNN's native exp/log/sqrt use. The fitter owns WHAT to approximate
// and the natural-basis coefficients; the kernel owns the intrinsic decompose +
// scale fold + recombine. The evaluator tag owns this standalone path; the
// coefficient CSV owns the reduced-domain fit.
//
// These are standalone evaluators: when one of RANGE_REDUCTION_*_HW is defined
// the per-tile loop calls the matching exp_hw_eval / log_hw_eval / pow_hw_eval
// directly and the piecewise segment cascade is bypassed entirely.
// ============================================================================

#if defined(EXPONENT_ALU_EXP2)
// exp(x) = 2^(x * log2e). Decompose x*log2e into integer i + fraction f via the
// hardware exponent ALU; 2^f via the fitter's degree-N poly; recombine setexp.
//
// The fitter emits NATURAL [0,1)-basis coeffs (EXP_HW_COEFFS, degree N) for
// g(f)=2^f. exman() returns the fraction scaled by 2^23, so we NORMALIZE it back
// to a float f in [0,1) (multiply by 2^-23) and run a plain degree-N Horner over
// the natural coeffs — exactly like pow_hw_eval normalizes its mantissa. (The old
// per-coeff 2^-23k fold was hardcoded to degree 2 and underflowed past degree 3.)
//
// The fitter tags EXP_HW_MULT = log2-domain multiplier (1.0 for 2^x, log2e for
// e^x, -log2e for exp(-x)) and an optional compose post-transform (minus_one for
// expm1, sigmoid for 1/(1+exp(-x))). Honor whatever the codegen emits.
#ifndef EXP_HW_MULT
#define EXP_HW_MULT 1.4426950216293334961f  // default 1/ln2 == log2(e)
#endif

#if defined(EXP_HW_FUSED)
// Per-coeff scale that folds the exman 2^23-normalize into the Horner
// coefficients: coeff at polynomial power k is multiplied by 2^-23k so the
// Horner can run over the UNNORMALIZED exman fraction (TTNN exp_21f trick).
// constexpr so the compiler bakes it in — zero runtime cost.
constexpr float exp_hw_fused_scale(int k) {
    float s = 1.0f;
    for (int i = 0; i < k; i++) {
        s *= 0x1p-23f;
    }
    return s;
}
#endif

template <uint32_t DEG>
inline vFloat exp_hw_eval(vFloat x) {
    constexpr float MULT = EXP_HW_MULT;

    vFloat xlog2 = x * MULT + 127.0f;

    // Full-range safety clamp: keep xlog2 in [0, 255] so the implicit float->int
    // conversion below cannot wrap (TTNN does this in its non-unsafe path).
    vFloat thr_lo = 0.0f;
    vFloat thr_hi = 255.0f;
    vec_min_max(thr_lo, xlog2);  // xlog2 = max(0, xlog2)
    vec_min_max(xlog2, thr_hi);  // xlog2 = min(xlog2, 255)

    // Branch-free float->int: shift mantissa left by (exp - bias) bits.
    vInt e = exexp(xlog2);
    vInt m = exman(xlog2, MantissaMode::ImplicitOne);
    m = shft(m, e, ShiftMode::Logical);
    vFloat z = as<vFloat>(m);

    vInt ep = exexp(z, ExponentMode::NoDebias);  // 2^(integer part)
    vMag fm = exman(z);                          // fraction * 2^23
    // Normalize the exman 2^23-scaled fraction back to a float f in [0,1).
    vFloat f = convert<vFloat>(fm, RoundMode::Nearest) * 0x1p-23f;

    // Plain degree-N Horner for 2^f over the natural [0,1) coeffs.
    const float* c = EXP_HW_COEFFS;
    vFloat p = c[DEG];
#pragma GCC unroll 16
    for (int k = (int)DEG - 1; k >= 0; k--) {
        p = p * f + c[k];
    }

    // Recombine 2^i * 2^f. `ep` is the biased exponent of the integer part
    // (== i + 127). setexp only REPLACES p's exponent field, keeping its
    // mantissa — which is correct only when p in [1,2) (exponent field 127).
    // The fitter's natural [0,1) fit makes g(0)=c[0] dip just below 1.0 for some
    // degrees (e.g. odd-degree exp2: c0=0.99992), putting p in [0.5,1) at f~=0
    // (exponent field 126). Replacing that with ep then over-scales by 2x. Add
    // p's own exponent deviation from the bias so the integer part composes with
    // p's actual magnitude (mirrors pow_hw_eval's setexp(s, s_exp + q)).
    vInt pe = exexp(p, ExponentMode::NoDebias);
    vFloat y = setexp(p, ep + pe - 127);  // 2^i * 2^f == base^x

    // Optional compose post-transform (fitter folds the activation around exp2).
#if defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
    // y == exp(-x); sigmoid(x) = 1 / (1 + exp(-x)).
#if defined(USE_BF16)
    y = ckernel::sfpu::sfpu_reciprocal_iter<1>(1.0f + y);
#else
    y = ckernel::sfpu::sfpu_reciprocal_iter<2>(1.0f + y);
#endif
#if defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
    y = x * y;
#endif
#elif defined(EXP_HW_COMPOSE_MINUS_ONE)
    // expm1(x) = exp(x) - 1.
    y = y - 1.0f;
#endif
    return y;
}

#if defined(HW_PRELOAD)
// ----------------------------------------------------------------------------
// GENERIC constant-pool preload variant of exp_hw_eval.
//
// Generalizes the proven exp deg-2 prototype: the codegen turns on HW_PRELOAD
// for EVERY exponent-ALU kind (exp2/log2/pow) at ANY degree (no per-activation
// hand-wiring). The mechanism is purely about WHERE the loop-invariant
// constexpr constants live so the recorded replay body re-loads none of them:
//
//   - vConstFloatPrgm0/1/2 : the 3 hottest constants, programmed ONCE before the
//     tile loop (persist across every replayed body + every tile).
//   - hoisted vFloat cv[]  : the remaining coefficients, declared as pre-loop
//     locals by the caller (hw_reduce) so GCC keeps them in iteration-invariant
//     LREGs — no SFPLOADI inside the recorded body.
//
// Per-kind ranking (deterministic, owned by the kernel since the kind dictates
// which constant is touched every element):
//   exp2 : prgm0 = MULT, prgm1 = c[DEG], prgm2 = c[DEG-1]; cv[] = c[DEG-2..0]
//          (255.0 clamp also hoisted; 0.0/127.0 are const-lane/SFPMAD imm).
// Anything past the prgm + LREG budget the compiler simply emits as in-body
// literals; the codegen logs that count (HW_PRELOAD_SPILL) — never silent.
//
// COMPOSE_SIGMOID and COMPOSE_SIGMOID_PRODUCT reserve vConstFloatPrgm0 for
// sfpu_reciprocal (it expects 2.0 there, set by sfpu_reciprocal_init). In that
// case MULT is demoted from prgm0 to a hoisted LREG param (mult_hoist) so the
// two uses of prgm0 never collide.
//
// The remaining coefficients (c[DEG-2..0]) are read from the constexpr global
// EXP_HW_COEFFS: the compiler hoists what fits in the LREG budget and emits the
// rest as in-body literals (the natural "spill" the codegen reports) — this
// avoids forcing a fixed-size vFloat array that would overflow the register file
// under heavy paths like sigmoid (reciprocal + clamp + mult all live at once).
//
// Math is byte-identical to exp_hw_eval (same SFPMAD Horner, same order).
// cvspill points at a caller-hoisted, loop-invariant vFloat array holding the
// coefficients BELOW the two prgm-resident ones: cvspill[0]=c[DEG-2],
// cvspill[1]=c[DEG-3], ... cvspill[DEG-2]=c[0]. Materializing them as pre-loop
// LREGs (not reading EXP_HW_COEFFS[k] inside the body) is FIX A: it drives the
// in-body SFPLOADI count for the Horner chain to ZERO (top-2 in prgm, rest in
// LREGs) instead of 2-per-spilled-coeff. nullptr -> fall back to the global.
template <uint32_t DEG>
inline vFloat exp_hw_eval_preloaded(
    vFloat x,
    vFloat thr_hi_hoist,
    vFloat c127_hoist,
    const vFloat* cvspill
#if defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
    ,
    vFloat mult_hoist
#endif
) {
    // MULT lives in prgm0 normally; under sigmoid compose prgm0 is owned by the
    // reciprocal, so MULT is read from the hoisted LREG instead. The bias 127.0
    // is supplied via a hoisted, loop-invariant register (c127_hoist) so the
    // compiler fuses x*MULT + 127 into ONE SFPMAD (matching TTNN exp_21f) rather
    // than emitting a separate SFPMUL + SFPADDI for the literal addend.
#if defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
    vFloat xlog2 = x * mult_hoist + c127_hoist;
#else
    vFloat xlog2 = x * vConstFloatPrgm0 + c127_hoist;
#endif

    // ------------------------------------------------------------------
    // INSTRUCTION-COUNT FAST PATH (matches TTNN's exp_21f schedule). Two
    // independent cuts, each gated by a generator-emitted macro so the generic
    // path stays correct for fits where the cut is unsafe:
    //   EXP_HW_FUSED       : fold the 2^-23 normalize into the coeffs (pre-scaled
    //                        in prgm1/prgm2/cvspill) -> removes one SFPMUL.
    //   EXP_HW_BARE_SETEXP : drop the pe correction -> removes one SFPEXEXP +
    //                        two SFPIADD. Valid when g(f) in [1,2) (c0>=1).
    // ------------------------------------------------------------------

    // Full-range safety clamp to [0, 255]. Two SFPSWAP (vec_min_max): the lower
    // max(0,xlog2) flushes inputs below ~-88 to 0, the upper min(xlog2,255)
    // prevents the implicit float->int conversion from wrapping. 0.0 uses the
    // const-0 lane (no SFPLOADI); 255.0 comes from the hoisted thr_hi_hoist.
    //
    // We measured the TTNN-style single-swap + masked-lower variants (v_if flush
    // and branch-free integer sign-mask); both cost MORE here than the two clean
    // swaps because the compiler can't hide the mask ops in SFPMAD windows the
    // way TTNN's hand-recorded TTI replay buffer does. The two swaps stay.
    vFloat thr_lo = 0.0f;
    vec_min_max(thr_lo, xlog2);        // xlog2 = max(0, xlog2)
    vec_min_max(xlog2, thr_hi_hoist);  // xlog2 = min(xlog2, 255)

    // Branch-free float->int: shift mantissa left by (exp - bias) bits.
    vInt e = exexp(xlog2);
    vInt m = exman(xlog2, MantissaMode::ImplicitOne);
    m = shft(m, e, ShiftMode::Logical);
    vFloat z = as<vFloat>(m);

    vInt ep = exexp(z, ExponentMode::NoDebias);  // 2^(integer part)
    vMag fm = exman(z);                          // fraction * 2^23

    // Fraction for the 2^f refinement polynomial.
#if defined(EXP_HW_FUSED)
    // FUSED: skip the `* 0x1p-23f` normalize. The coeffs in prgm1/prgm2/cvspill
    // are pre-scaled by 2^-23k (generator), so Horner runs over the raw exman
    // fraction. Removes one SFPMUL per element (TTNN exp_21f trick).
    vFloat f = convert<vFloat>(fm, RoundMode::Nearest);
#else
    vFloat f = convert<vFloat>(fm, RoundMode::Nearest) * 0x1p-23f;
#endif

    // Degree-N Horner. Top-2 coeffs come from prgm regs; the rest from the
    // caller-hoisted LREG array cvspill (FIX A: no in-body SFPLOADI for them).
    // p = (((c[DEG])*f + c[DEG-1])*f + c[DEG-2])*f + ... + c[0]
    vFloat p;
    if constexpr (DEG >= 1) {
        p = vConstFloatPrgm1 * f + vConstFloatPrgm2;  // c[DEG]*f + c[DEG-1]
#pragma GCC unroll 16
        for (int k = (int)DEG - 2; k >= 0; k--) {
            // cvspill index walks c[DEG-2], c[DEG-3], ... c[0] as k descends.
            p = p * f + cvspill[(int)DEG - 2 - k];
        }
    } else {
        p = vConstFloatPrgm1;  // degree-0: c[0]
    }

    // Recombine 2^i * 2^f.
#if defined(EXP_HW_BARE_SETEXP)
    // BARE: p is in [1,2) by construction (rising-to-2 exp2 fit, c0>=1), so a
    // plain setexp(p, ep) is correct. The lower bound (negative-input flush) is
    // already enforced by the max(0,xlog2) swap above. Removes one SFPEXEXP +
    // two SFPIADD vs the pe-corrected form below (TTNN exp_21f recombine).
    vFloat y = setexp(p, ep);
#else
    // setexp replaces p's exponent but keeps its mantissa, correct only when p
    // in [1,2). The fitter's g(0)=c[0] dips below 1.0 for some degrees
    // (odd-degree exp2 c0=0.99992), so p lands in [0.5,1) at f~=0 and a bare
    // setexp(p, ep) over-scales by 2x. Add p's own exponent deviation from the
    // bias (matches the non-preloaded exp_hw_eval + pow path).
    vInt pe = exexp(p, ExponentMode::NoDebias);
    vFloat y = setexp(p, ep + pe - 127);  // 2^i * 2^f == base^x
#endif

#if defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
#if defined(USE_BF16)
    y = ckernel::sfpu::sfpu_reciprocal_iter<1>(1.0f + y);
#else
    y = ckernel::sfpu::sfpu_reciprocal_iter<2>(1.0f + y);
#endif
#if defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
    y = x * y;
#endif
#elif defined(EXP_HW_COMPOSE_MINUS_ONE)
    y = y - 1.0f;
#endif
    return y;
}
#endif  // HW_PRELOAD
#endif

#if defined(EXPONENT_ALU_LOG2)
// log2(x) = e + log2(m) for x = 2^e * m, m in [1,2). The fitter fits h(m)=log2(m)
// on [1,2) and emits NATURAL [1,2]-basis coeffs (LOG_HW_C0..); the kernel
// extracts e (integer log2, free via exexp) and m (exman/setexp), evaluates the
// poly, adds e, and applies an optional base scale (ln2 for log, log10(2) for
// log10) via LOG_HW_SCALE.
#ifndef LOG_HW_SCALE
#define LOG_HW_SCALE 1.0f
#endif
// log1p(x) = log(x + 1): the fitter tags expalu_input_offset = 1.0 so the log2
// decompose operates on (x + offset). Default 0.0 keeps plain log/log2/log10.
#ifndef LOG_HW_INPUT_OFFSET
#define LOG_HW_INPUT_OFFSET 0.0f
#endif
template <uint32_t DEG>
inline vFloat log_hw_eval(vFloat x) {
    vFloat x_in = x;  // original (used for special-case tests)
    constexpr float OFFSET = LOG_HW_INPUT_OFFSET;
    vFloat xd = (OFFSET != 0.0f) ? (x + OFFSET) : x;

    // Extract biased exponent e and mantissa m in [1,2) of the decompose input.
    vInt biased = exexp(xd, ExponentMode::NoDebias);
    vInt e_int = biased - 127;
    vFloat m = setexp(xd, 127);

    // Horner h(m)=log2(m). The fitter tags the polynomial basis: the natural
    // [1,2] basis evaluates in m, the m_minus_1 basis (the codegen default for
    // log2) evaluates in (m-1) so c0==0 and h(1)==0 exactly.
#ifdef LOG_HW_BASIS_M_MINUS_1
    vFloat u = m - 1.0f;
#else
    vFloat u = m;
#endif
    const float* c = LOG_HW_COEFFS;
    vFloat h = c[DEG];
#pragma GCC unroll 16
    for (int k = (int)DEG - 1; k >= 0; k--) {
        h = h * u + c[k];
    }

    // e + h(m); int32_to_float wants sign-magnitude for negatives.
    v_if(e_int < 0) { e_int = setsgn(~e_int + 1, 1); }
    v_endif;
    vFloat e_float = int32_to_float(e_int, RoundMode::Nearest);
    vFloat result = (e_float + h) * LOG_HW_SCALE;

    // Special cases on the DECOMPOSE input (x + offset): log(0) = -inf, log(neg) = NaN.
    // For log1p this is the (x + 1) singularity at x = -1.
    v_if(xd < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(xd == 0.0f) { result = -std::numeric_limits<float>::infinity(); }
    v_endif;
    (void)x_in;
    return result;
}

#if defined(HW_PRELOAD)
// GENERIC constant-pool preload variant of log_hw_eval (any degree).
// Ranking: prgm0 = LOG_HW_SCALE, prgm1 = c[DEG], prgm2 = c[DEG-1]; cv[] = c[DEG-2..0].
// 127 / 1.0 are SFPMAD imm / const-lane. Math byte-identical to log_hw_eval.
// cvspill: caller-hoisted LREG array of the below-prgm coeffs (c[DEG-2..0]).
template <uint32_t DEG>
inline vFloat log_hw_eval_preloaded(vFloat x, const vFloat* cvspill) {
    constexpr float OFFSET = LOG_HW_INPUT_OFFSET;
    vFloat xd = (OFFSET != 0.0f) ? (x + OFFSET) : x;

    vInt biased = exexp(xd, ExponentMode::NoDebias);
    vInt e_int = biased - 127;
    vFloat m = setexp(xd, 127);

#ifdef LOG_HW_BASIS_M_MINUS_1
    vFloat u = m - 1.0f;
#else
    vFloat u = m;
#endif
    vFloat h;
    if constexpr (DEG >= 1) {
        h = vConstFloatPrgm1 * u + vConstFloatPrgm2;  // c[DEG]*u + c[DEG-1]
#pragma GCC unroll 16
        for (int k = (int)DEG - 2; k >= 0; k--) {
            h = h * u + cvspill[(int)DEG - 2 - k];  // c[DEG-2], c[DEG-3], ... c[0]
        }
    } else {
        h = vConstFloatPrgm1;  // degree-0: c[0]
    }

    v_if(e_int < 0) { e_int = setsgn(~e_int + 1, 1); }
    v_endif;
    vFloat e_float = int32_to_float(e_int, RoundMode::Nearest);
    vFloat result = (e_float + h) * vConstFloatPrgm0;  // * LOG_HW_SCALE

    v_if(xd < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(xd == 0.0f) { result = -std::numeric_limits<float>::infinity(); }
    v_endif;
    return result;
}
#endif  // HW_PRELOAD
#endif

#if defined(EVAL_METHOD_NEWTON_ROOT)
// ============================================================================
// Newton-Raphson magic-seed square root (mirrors TTNN's native ckernel sqrt).
//
// This is the INTEGER-ROOT fast path. Instead of decomposing x = 2^e * m and
// rebuilding the exponent (the pow_hw path, ~39 SFPU instrs), we operate
// directly on the raw IEEE bits:
//
//   i  = bits(x) >> 1                                  (one shift)
//   y0 = bits( MAGIC - i )                             (one integer subtract)
//
// The magic-constant subtraction produces a seed y0 ~= 1/sqrt(x)-ish whose
// EXPONENT is already halved AND whose parity (the sqrt(2) odd-exponent scale)
// is already folded into the seed -- so there is NO exexp/setexp/cast/parity
// cascade at all. Two Newton-Raphson refinement steps recover full fp32
// precision. The 3 loop-invariant constants (MAGIC seed, two Newton coeffs)
// live in the programmable const registers (preloaded once), so the recorded
// per-element body reloads NONE of them.
//
// Algorithm: SQRT_23-bits from Kokosinski et al. (2024), the exact algorithm
// the native blackhole ckernel_sfpu_sqrt.h uses. The fitter owns the seed
// magic and Newton coefficients (metadata: newton_root_magic / newton_root_c1
// / newton_root_c2), so this generalizes to any even-root / accuracy target it
// chooses to tune -- it is not hand-wired for sqrt.
#ifndef NEWTON_ROOT_MAGIC
#define NEWTON_ROOT_MAGIC 0x5f1110a0
#endif
#ifndef NEWTON_ROOT_C1
#define NEWTON_ROOT_C1 2.2825186f
#endif
#ifndef NEWTON_ROOT_C2
#define NEWTON_ROOT_C2 2.2533049f
#endif
#ifndef NEWTON_ROOT_C0
#define NEWTON_ROOT_C0 0x1.c09806p0f
#endif
// Root order N (2 = sqrt/rsqrt, 3 = cbrt) and the rsqrt/inverse flavour are
// metadata-driven (newton_root_n / newton_root_reciprocal). Defaults keep the
// path byte-identical to native sqrt when no extra tags are emitted.
#ifndef NEWTON_ROOT_N
#define NEWTON_ROOT_N 2
#endif
#ifndef NEWTON_ROOT_ITERS
#define NEWTON_ROOT_ITERS 3
#endif

#if (NEWTON_ROOT_N == 2) && !defined(NEWTON_ROOT_RECIPROCAL)
// --- sqrt: magic seed + SQRT_23-bit double-Newton (native parity) ------------
inline vFloat newton_root_sqrt(vFloat x) {
    vInt i = reinterpret<vInt>(reinterpret<vUInt>(x) >> 1);
    vFloat y = reinterpret<vFloat>(vConstIntPrgm0 - i);  // MAGIC seed (preloaded)

    // SQRT_23-bits: two Newton-Raphson steps. prgm1/prgm2 preloaded.
    vFloat xy = x * y;
    vFloat negative_y = -y;
    vFloat c = negative_y * xy;
    y = y * (vConstFloatPrgm1 + c * (vConstFloatPrgm2 + c));
    xy = x * y;
    negative_y = -y;
    vFloat one_minus_xyy = vConst1 + (negative_y * xy);
    vFloat half_xy = addexp(xy, -1);  // 0.5*xy via exponent decrement (no 0.5 immediate)
    vFloat infinity = sFloat16b(std::numeric_limits<float>::infinity());
    // Skip the final correction at x==inf (avoids inf-inf = NaN; y already inf).
    v_if(reinterpret<vInt>(x) < reinterpret<vInt>(infinity)) { y = one_minus_xyy * half_xy + xy; }
    v_endif;
    // sqrt of a negative is NaN.
    v_if(x < 0.0f) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;
    return y;
}
#endif  // sqrt variant

#if (NEWTON_ROOT_N == 2) && defined(NEWTON_ROOT_RECIPROCAL)
// --- rsqrt: classic inverse-sqrt magic seed + Newton (y = y*(1.5 - 0.5*x*y*y))-
// The seed IS 1/sqrt(x) directly, so NO final reciprocal op is needed. Each
// Newton step is one mul + two mads. 2 steps -> <=1 bf16 ULP, 3 -> exact.
inline vFloat newton_root_rsqrt(vFloat x) {
    vInt i = reinterpret<vInt>(reinterpret<vUInt>(x) >> 1);
    vFloat y = reinterpret<vFloat>(vConstIntPrgm0 - i);  // inverse-sqrt MAGIC seed
    vFloat half_x = addexp(x, -1);                       // 0.5*x (exponent decrement)
#pragma GCC unroll 4
    for (int s = 0; s < NEWTON_ROOT_ITERS; s++) {
        y = y * (vConstFloatPrgm1 - half_x * (y * y));  // prgm1 = 1.5
    }
    v_if(x < 0.0f) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;
    v_if(x == 0.0f) { y = std::numeric_limits<float>::infinity(); }
    v_endif;
    return y;
}
#endif  // rsqrt variant

#if (NEWTON_ROOT_N == 3)
#if defined(NEWTON_ROOT_ALGORITHM_CBRT_MAGIC)
// --- cbrt: native-style Moroz magic seed + correction ----------------------
//
// Metadata declares this algorithm through NEWTON_ROOT_ALGORITHM_CBRT_MAGIC.
// The body mirrors ckernel_sfpu_cbrt.h generically enough for any lowering that
// supplies the same payload: root_n=3, seed magic, seed scale, and correction
// coefficients. It does not inspect the activation name.
#ifndef NEWTON_ROOT_MAGIC_SCALE
#define NEWTON_ROOT_MAGIC_SCALE 256.0f
#endif
#ifndef NEWTON_ROOT_MAGIC_BIAS
#define NEWTON_ROOT_MAGIC_BIAS 8388608.0f
#endif
#ifndef NEWTON_ROOT_NEG_INV_N_SCALED
#define NEWTON_ROOT_NEG_INV_N_SCALED -0x1.555556p-10f
#endif
inline vFloat newton_root_cbrt_magic(vFloat x) {
    vFloat ax = setsgn(x, 0);

    vFloat f = convert<vFloat>(as<vSMag>(ax), RoundMode::Nearest);
    vFloat magic = ((float)NEWTON_ROOT_MAGIC) / NEWTON_ROOT_MAGIC_SCALE + NEWTON_ROOT_MAGIC_BIAS;
    f = f * NEWTON_ROOT_NEG_INV_N_SCALED + magic;

    vFloat y = as<vFloat>(as<vInt>(f) << 8);

#ifdef USE_BF16
    vFloat d = ax * (y * y);
    vFloat c = d * y;
    vFloat t = c * (vConstFloatPrgm2 * c + vConstFloatPrgm1) + vConstFloatPrgm0;
    d = copysgn(d, x);
    y = d * (t * t);
#else
    vFloat c = (ax * y) * (y * y);
    y = y * (c * (vConstFloatPrgm2 * c + vConstFloatPrgm1) + vConstFloatPrgm0);

    vFloat d = ax * (y * y);
    c = d * y + vConstNeg1;
    vFloat negative_third = addexp(vFloat(NEWTON_ROOT_NEG_INV_N_SCALED), 8);
    vFloat t = c * negative_third + vConst1;
    d = copysgn(d, x);
    y = d * (t * t);
#endif
    return y;
}
#else
// --- cbrt: minimal exponent seed + DIVISION-FREE cubic Householder ---------
// Odd function: work on |x|, restore sign. The SFPU has no 32-bit float->int (so
// the classic bits/3+magic seed isn't expressible) AND its reciprocal LLK is
// register-heavy (inlining it per-element overflows the SFPU register file ->
// trisc "maximum reload insns" ICE). So we iterate the INVERSE cube root
// w -> |x|^(-1/3) with a DIVISION-FREE cubic Householder step:
//     c = 1 - |x|*w^3 ;  w <- w * (1 + c/3 + 2c^2/9)
// then cbrt(|x|) = |x| * w^2. Seed w0 = 2^(-round(e/3)) * (-0.27*m + 1.25), a
// linear m^(-1/3) approx. The exponent-parity is NOT folded in (keeps the body
// register-light); 3 cubic steps recover <=1 bf16 ULP (validated).
inline vFloat newton_root_cbrt(vFloat x) {
    vInt sign_bits = reinterpret<vInt>(x) & (vInt)0x80000000;
    vFloat ax = setsgn(x, 0);  // work on |x|; cbrt is odd, sign restored at end

    vInt e_int = exexp(ax, ExponentMode::NoDebias) - 127;
    vFloat m = setexp(ax, 127);

    // q = round(e/3) via the 0x4B400000 float magic (e small -> exact in float).
    // Round-to-nearest (NOT floor+parity): keeps the body register-light enough
    // to avoid the trisc "maximum reload insns" ICE. The missing exponent parity
    // is absorbed by one extra cubic Householder step.
    const vFloat magic = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat ef = int32_to_float(e_int, RoundMode::Nearest);
    v_if(e_int < 0) { ef = -int32_to_float(~e_int + 1, RoundMode::Nearest); }
    v_endif;
    vInt q = reinterpret<vInt>(ef * (1.0f / 3.0f) + magic) - reinterpret<vInt>(magic);

    // w0 = 2^(-q) * (-0.27*m + 1.25): linear m^(-1/3) seed, exponent shifted -q.
    vFloat wm = -0.27f * m + 1.25f;
    vFloat w = setexp(wm, exexp(wm, ExponentMode::NoDebias) - q);

    // Division-free cubic Householder on the inverse cube root.
    const vFloat a13 = 1.0f / 3.0f;
    const vFloat a29 = 2.0f / 9.0f;
    for (int s = 0; s < NEWTON_ROOT_ITERS; s++) {
        vFloat c = 1.0f - ax * (w * w * w);
        w = w * (1.0f + c * a13 + (c * c) * a29);
    }
    vFloat y = ax * w * w;  // cbrt(|x|) = |x| * w^2
    // restore sign (cbrt is odd); cbrt(0)=0 handled explicitly.
    y = reinterpret<vFloat>(reinterpret<vInt>(y) | sign_bits);
    v_if(x == 0.0f) { y = 0.0f; }
    v_endif;
    return y;
}
#endif
#endif  // cbrt variant

template <uint32_t DEG>
inline vFloat newton_root_eval(vFloat x) {
    (void)DEG;  // degree is irrelevant for the Newton path (kept for caller symmetry)
#if (NEWTON_ROOT_N == 3)
#if defined(NEWTON_ROOT_ALGORITHM_CBRT_MAGIC)
    return newton_root_cbrt_magic(x);
#else
    return newton_root_cbrt(x);
#endif
#elif defined(NEWTON_ROOT_RECIPROCAL)
    return newton_root_rsqrt(x);
#else
    return newton_root_sqrt(x);
#endif
}
#endif

#if defined(EVAL_METHOD_ASIN_ACOS)
inline vFloat asin_acos_sqrt(vFloat x) {
    vInt i = reinterpret<vInt>(reinterpret<vUInt>(x) >> 1);
    vFloat y = reinterpret<vFloat>(0x5f1110a0 - i);

    vFloat xy = x * y;
    vFloat c = -y * xy;
    y = y * (2.2825186f + c * (2.2533049f + c));
    xy = x * y;
    vFloat one_minus_xyy = vConst1 + (-y * xy);
    vFloat half_xy = addexp(xy, -1);
    vFloat infinity = sFloat16b(std::numeric_limits<float>::infinity());
    v_if(reinterpret<vInt>(x) < reinterpret<vInt>(infinity)) { y = one_minus_xyy * half_xy + xy; }
    v_endif;
    return y;
}

inline vFloat asin_acos_ratio_poly(vFloat z) {
    vFloat u = z * z;
    vFloat ratio = ASIN_ACOS_COEFFS[ASIN_ACOS_DEGREE];
#pragma GCC unroll 8
    for (int k = ASIN_ACOS_DEGREE - 1; k >= 0; --k) {
        ratio = ratio * u + ASIN_ACOS_COEFFS[k];
    }
    return z * ratio;
}

inline vFloat asin_acos_eval(vFloat x) {
    vFloat ax = setsgn(x, 0);
    vFloat asin_abs = ASIN_ACOS_PI_2;

    v_if(ax <= ASIN_ACOS_DIRECT_THRESHOLD) { asin_abs = asin_acos_ratio_poly(ax); }
    v_else {
        vFloat t = (vConst1 - ax) * 0.5f;
        vFloat root = asin_acos_sqrt(t);
        asin_abs = ASIN_ACOS_PI_2 - 2.0f * asin_acos_ratio_poly(root);
    }
    v_endif;

    vFloat asin_signed = copysgn(asin_abs, x);
#if defined(ASIN_ACOS_OP_ACOS)
    return ASIN_ACOS_PI_2 - asin_signed;
#else
    return asin_signed;
#endif
}
#endif

#if defined(EXPONENT_ALU_POW)
// pow path for sqrt/rsqrt/cbrt. For root order N (POW_HW_ROOT_N) and x = 2^e * m
// with m in [1,2): root_N(x) = 2^(e/N) * root_N(2^r) * root_N(m), where
// e = N*q + r, r in {0..N-1}. The fitter fits p(m)=root_N(m) on [1,2)
// (NATURAL [1,2]-basis coeffs POW_HW_COEFFS); the kernel splits e into q (folded
// into the exponent via setexp) and parity remainder r (selects the scale
// constant POW_HW_SCALE_C{r} = root_N(2^r)), evaluates the poly, recombines.
//
// Odd roots (cbrt, N=3) are odd functions: cbrt(-x) = -cbrt(x). The kernel
// strips the sign (works on |x|) and restores it on the result. Even roots
// (sqrt, N=2) of negatives are NaN.
//
// expalu_reciprocal (POW_HW_RECIPROCAL) folds the final 1/result for rsqrt
// (= 1/sqrt(x)): one Newton-Raphson reciprocal applied AFTER recombine.
//
// Backward-compatible defaults: N=2, scales = {1, sqrt(2)}.
#ifndef POW_HW_ROOT_N
#define POW_HW_ROOT_N 2
#endif
#ifndef POW_HW_SCALE_C0
#define POW_HW_SCALE_C0 1.0f
#endif
#ifndef POW_HW_SCALE_C1
#define POW_HW_SCALE_C1 1.4142135623730951f  // sqrt(2)
#endif
// C2 only consulted when ROOT_N >= 3 (cbrt). Default cbrt(4) is harmless otherwise.
#ifndef POW_HW_SCALE_C2
#define POW_HW_SCALE_C2 1.5874010519681994f  // cbrt(4)
#endif
template <uint32_t DEG>
inline vFloat pow_hw_eval(vFloat x) {
    constexpr int ROOT_N = POW_HW_ROOT_N;
    constexpr bool ODD_ROOT = (ROOT_N % 2 == 1);

    // Odd roots act on |x|; sign restored at the end. Even roots: neg -> NaN.
    vInt sign_bits = reinterpret<vInt>(x) & (vInt)0x80000000;
    vFloat ax = ODD_ROOT ? setsgn(x, 0) : x;

    // Extract biased exponent e and mantissa m in [1,2).
    vInt biased = exexp(ax, ExponentMode::NoDebias);
    vInt e_int = biased - 127;
    vFloat m = setexp(ax, 127);

    // Horner p(m)=root_N(m) in the natural [1,2] basis.
    const float* c = POW_HW_COEFFS;
    vFloat s = c[DEG];
#pragma GCC unroll 16
    for (int k = (int)DEG - 1; k >= 0; k--) {
        s = s * m + c[k];
    }

    // e = N*q + r, r in {0..N-1}. Compute q = floor(e/N) and r via float arith
    // (e is small, exact in float), matching the cbrt software path's structure.
    const vFloat magic = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat ef = int32_to_float(e_int, RoundMode::Nearest);
    v_if(e_int < 0) {
        vInt mag = ~e_int + 1;
        ef = -int32_to_float(mag, RoundMode::Nearest);
    }
    v_endif;

    vInt q;
    vInt r;
    if constexpr (ROOT_N == 2) {
        r = e_int & 1;  // two's-complement low bit is parity for both signs
        vFloat qf = (ef - int32_to_float(r, RoundMode::Nearest)) * 0.5f;
        q = reinterpret<vInt>(qf + magic) - reinterpret<vInt>(magic);
    } else {
        // General floor-divide by N via round-to-nearest of e/N then correct r.
        constexpr float INV_N = 1.0f / (float)ROOT_N;
        vFloat q_approx = ef * INV_N;
        vFloat q_rounded = q_approx + magic;
        q = reinterpret<vInt>(q_rounded) - reinterpret<vInt>(magic);
        vFloat q_back = q_rounded - magic;
        // r = e - N*q (in float, exact), then to int via magic.
        vFloat nq = q_back;
#pragma GCC unroll 8
        for (int i = 1; i < ROOT_N; i++) {
            nq = nq + q_back;
        }  // nq = N*q_back
        vFloat r_float = ef - nq;
        r = reinterpret<vInt>(r_float + magic) - reinterpret<vInt>(magic);
        // Normalize r into [0, N-1], adjusting q accordingly.
        v_if(r < 0) {
            q = q - 1;
            r = r + ROOT_N;
        }
        v_endif;
        v_if(r >= ROOT_N) {
            q = q + 1;
            r = r - ROOT_N;
        }
        v_endif;
    }

    // Apply scale root_N(2^r) for the parity remainder r.
    v_if(r == 1) { s = s * POW_HW_SCALE_C1; }
    v_endif;
    if constexpr (ROOT_N >= 3) {
        v_if(r == 2) { s = s * POW_HW_SCALE_C2; }
        v_endif;
    }

    // Multiply by 2^q via exponent addition.
    vInt s_exp = exexp(s, ExponentMode::NoDebias);
    vFloat result = setexp(s, s_exp + q);

#if defined(POW_HW_RECIPROCAL)
    // rsqrt etc.: final 1/result. Newton-Raphson reciprocal (3 iters, no init dep).
    result = ckernel::sfpu::sfpu_reciprocal_iter<3>(result);
#endif

    if constexpr (ODD_ROOT) {
        // Restore original sign (root is an odd function): cbrt(-x) = -cbrt(x).
        result = reinterpret<vFloat>(reinterpret<vInt>(result) | sign_bits);
        // x == 0 -> 0 (sign already carried; |0| poly path is finite).
        v_if(ax == 0.0f) { result = reinterpret<vFloat>(sign_bits); }
        v_endif;
    } else {
        // Special cases: even root of negative is NaN; root(0)=0.
        v_if(x < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(x == 0.0f) {
#if defined(POW_HW_RECIPROCAL)
            result = std::numeric_limits<float>::infinity();  // rsqrt(0) = +inf
#else
            result = 0.0f;  // sqrt(0) = 0
#endif
        }
        v_endif;
    }
    return result;
}

#if defined(HW_PRELOAD)
// GENERIC constant-pool preload variant of pow_hw_eval (any degree).
// Ranking: prgm0 = SQRT2, prgm1 = c[DEG], prgm2 = c[DEG-1]; cv[] = c[DEG-2..0].
// 'magic' (round helper) is hoisted into an LREG by the caller and passed by value.
// Math byte-identical to pow_hw_eval.
// cvspill: caller-hoisted LREG array of the below-prgm coeffs (c[DEG-2..0]).
template <uint32_t DEG>
inline vFloat pow_hw_eval_preloaded(vFloat x, vFloat magic_hoist, const vFloat* cvspill) {
    // The preload optimization (prgm0 = sqrt(2)) only applies to plain sqrt
    // (ROOT_N == 2, no reciprocal). For odd roots (cbrt) and the reciprocal
    // fold (rsqrt) the scale/sign/recip logic doesn't fit the 3-prgm budget, so
    // fall back to the fully-general evaluator — those kernels are correctness-
    // first and still avoid the segment cascade.
#if (POW_HW_ROOT_N != 2) || defined(POW_HW_RECIPROCAL)
    (void)magic_hoist;
    (void)cvspill;
    return pow_hw_eval<DEG>(x);
#else
    vInt biased = exexp(x, ExponentMode::NoDebias);
    vInt e_int = biased - 127;
    vFloat m = setexp(x, 127);

    vFloat s;
    if constexpr (DEG >= 1) {
        s = vConstFloatPrgm1 * m + vConstFloatPrgm2;  // c[DEG]*m + c[DEG-1]
#pragma GCC unroll 16
        for (int k = (int)DEG - 2; k >= 0; k--) {
            s = s * m + cvspill[(int)DEG - 2 - k];  // c[DEG-2], c[DEG-3], ... c[0]
        }
    } else {
        s = vConstFloatPrgm1;  // degree-0: c[0]
    }

    vInt r = e_int & 1;
    vFloat ef = int32_to_float(e_int, RoundMode::Nearest);
    v_if(e_int < 0) {
        vInt mag = ~e_int + 1;
        ef = -int32_to_float(mag, RoundMode::Nearest);
    }
    v_endif;
    vFloat qf = (ef - int32_to_float(r, RoundMode::Nearest)) * 0.5f;
    vInt q = reinterpret<vInt>(qf + magic_hoist) - reinterpret<vInt>(magic_hoist);

    // Apply sqrt(2) when e is odd (preloaded in vConstFloatPrgm0).
    v_if(r == 1) { s = s * vConstFloatPrgm0; }
    v_endif;

    vInt s_exp = exexp(s, ExponentMode::NoDebias);
    vFloat result = setexp(s, s_exp + q);

    v_if(x < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(x == 0.0f) { result = 0.0f; }
    v_endif;
    return result;
#endif
}
#endif  // HW_PRELOAD
#endif

// ============================================================================
// Asymptotic factoring: inline Cody-Waite exp for dominant factor computation
// Used when correction polynomials are multiplied by exp-based dominant factors
// to handle extreme dynamic range in function tails (e.g., GELU, sigmoid).
// Mutually exclusive with range reduction (never combined).
// ============================================================================
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || \
    defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR)

// Inline Cody-Waite exp(arg) — no overflow/underflow checks needed because
// the asymptotic region has known bounded input range.
// Degree-5 Taylor for exp(r), |r| < ln(2)/2: relative error < 3.3e-9.
inline vFloat asymptotic_exp(vFloat arg) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    vFloat z = arg * INV_LN2;

    // Round to nearest integer (branch-free magic-number technique)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat k = tmp - c231;
    vInt k_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);

    // Cody-Waite extended precision: r = arg - k*ln2
    constexpr float NEG_LN2_HI = -0.6931152343750000f;
    constexpr float NEG_LN2_LO = -3.19461832987e-05f;
    vFloat r = k * NEG_LN2_HI + arg;
    r = k * NEG_LN2_LO + r;

    // Degree-5 Taylor for exp(r)
    vFloat p = 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;

    // Scale by 2^k via exponent bit manipulation
    vInt p_exp = exexp_nodebias(p);
    return setexp(p, p_exp + k_int);
}

#endif  // ASYMPTOTIC_FACTOR_EXP_*

// Generic polynomial evaluation using Horner's method with compile-time optimization
template <uint32_t DEGREE>
inline vFloat eval_polynomial(const float* coeffs, vFloat x) {
    if constexpr (DEGREE == 0) {
        // Constant: y = c0
        return coeffs[0];
    } else if constexpr (DEGREE == 1) {
        // Linear: y = c0 + c1*x
        return coeffs[0] + coeffs[1] * x;
    } else if constexpr (DEGREE == 2) {
        // Quadratic: y = c0 + c1*x + c2*x²
        // Horner: (c2*x + c1)*x + c0
        return (coeffs[2] * x + coeffs[1]) * x + coeffs[0];
    } else if constexpr (DEGREE == 3) {
        // Cubic: y = c0 + c1*x + c2*x² + c3*x³
        // Horner: ((c3*x + c2)*x + c1)*x + c0
        return ((coeffs[3] * x + coeffs[2]) * x + coeffs[1]) * x + coeffs[0];
    } else if constexpr (DEGREE == 4) {
        // Quartic: y = c0 + c1*x + c2*x² + c3*x³ + c4*x⁴
        // Horner: ((((c4*x + c3)*x + c2)*x + c1)*x + c0
        vFloat result = coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 5) {
        // Quintic: y = c0 + ... + c5*x⁵
        vFloat result = coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 6) {
        // Hexic: y = c0 + ... + c6*x⁶
        vFloat result = coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 7) {
        // Septic: y = c0 + ... + c7*x⁷
        vFloat result = coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 8) {
        // Octic: y = c0 + ... + c8*x⁸
        vFloat result = coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 9) {
        // Nonic: y = c0 + ... + c9*x⁹
        vFloat result = coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 10) {
        // Decic: y = c0 + ... + c10*x¹⁰
        vFloat result = coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 11) {
        // Undecic: y = c0 + ... + c11*x¹¹
        vFloat result = coeffs[11];
        result = result * x + coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 12) {
        // Dodecic: y = c0 + ... + c12*x¹²
        vFloat result = coeffs[12];
        result = result * x + coeffs[11];
        result = result * x + coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 13) {
        // Tridecic: y = c0 + ... + c13*x¹³
        vFloat result = coeffs[13];
        result = result * x + coeffs[12];
        result = result * x + coeffs[11];
        result = result * x + coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 14) {
        // Tetradecic: y = c0 + ... + c14*x¹⁴
        vFloat result = coeffs[14];
        result = result * x + coeffs[13];
        result = result * x + coeffs[12];
        result = result * x + coeffs[11];
        result = result * x + coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 15) {
        // Pentadecic: y = c0 + ... + c15*x¹⁵
        vFloat result = coeffs[15];
        result = result * x + coeffs[14];
        result = result * x + coeffs[13];
        result = result * x + coeffs[12];
        result = result * x + coeffs[11];
        result = result * x + coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 16) {
        // Hexadecic: y = c0 + ... + c16*x¹⁶
        vFloat result = coeffs[16];
        result = result * x + coeffs[15];
        result = result * x + coeffs[14];
        result = result * x + coeffs[13];
        result = result * x + coeffs[12];
        result = result * x + coeffs[11];
        result = result * x + coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 32) {
        // Duotrigesic: y = c0 + ... + c32*x³²
        vFloat result = coeffs[32];
        for (int i = 31; i >= 0; i--) {
            result = result * x + coeffs[i];
        }
        return result;
    }
}

// x²-Horner evaluation for polynomials with parity structure.
// Halves Horner step count by evaluating in x² basis with stride-2 coefficient access.
//
// Odd parity (c0=c2=c4=...=0):  P(x) = x * Horner([c1,c3,c5,...], x²)
// Even parity (c1=c3=c5=...=0): P(x) = Horner([c0,c2,c4,...], x²)
//
// DEGREE is the nominal (max) degree — the LUT stores DEGREE+1 coefficients per segment,
// but zero-valued coefficients are skipped via stride-2 access.
#if defined(POLY_PARITY_ODD)
template <uint32_t DEGREE>
inline vFloat eval_polynomial_parity(const float* coeffs, vFloat x, vFloat x2) {
    // Odd: only odd-index coefficients are nonzero (c1,c3,c5,...)
    // TOP = highest odd index ≤ DEGREE
    constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = (TOP - 1) / 2;  // number of FMA steps after init
    vFloat result = coeffs[TOP];
#pragma GCC unroll 16
    for (int k = 1; k <= STEPS; k++) {
        result = result * x2 + coeffs[TOP - 2 * k];
    }
    return result * x;  // final *x for odd parity
}
#elif defined(POLY_PARITY_EVEN)
template <uint32_t DEGREE>
inline vFloat eval_polynomial_parity(const float* coeffs, vFloat x, vFloat x2) {
    // Even: only even-index coefficients are nonzero (c0,c2,c4,...)
    // TOP = highest even index ≤ DEGREE
    constexpr int TOP = (DEGREE % 2 == 0) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = TOP / 2;  // number of FMA steps after init
    vFloat result = coeffs[TOP];
#pragma GCC unroll 16
    for (int k = 1; k <= STEPS; k++) {
        result = result * x2 + coeffs[TOP - 2 * k];
    }
    return result;
}
#endif

#ifdef USE_DUAL_EVAL
// Dual polynomial evaluation using Horner's method with interleaved operations.
// Evaluates two independent x-vectors simultaneously to exploit SFPU instruction-level
// parallelism, hiding the 2-cycle SFPMAD latency by interleaving independent chains.
//
// Usage: eval_polynomial_dual<DEGREE>(coeffs, x1, x2, result1, result2)
// The strict alternation (result1 op, result2 op, result1 op, ...) ensures
// the compiler sees independent computation chains that can be pipelined.
template <uint32_t DEGREE>
inline void eval_polynomial_dual(const float* coeffs, vFloat x1, vFloat x2, vFloat& result1, vFloat& result2) {
    // Each coefficient is hoisted into a named vFloat so the compiler emits one sfploadi
    // pair and reuses the register as src2 for BOTH MADs, halving coefficient load count.
    if constexpr (DEGREE == 0) {
        vFloat c = coeffs[0];
        result1 = c;
        result2 = c;
    } else if constexpr (DEGREE == 1) {
        vFloat c1 = coeffs[1];
        vFloat c0 = coeffs[0];
        result1 = c0 + c1 * x1;
        result2 = c0 + c1 * x2;
    } else if constexpr (DEGREE == 2) {
        {
            vFloat c = coeffs[2];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 3) {
        {
            vFloat c = coeffs[3];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 4) {
        {
            vFloat c = coeffs[4];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 5) {
        {
            vFloat c = coeffs[5];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 6) {
        {
            vFloat c = coeffs[6];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 7) {
        {
            vFloat c = coeffs[7];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 8) {
        {
            vFloat c = coeffs[8];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 9) {
        {
            vFloat c = coeffs[9];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 10) {
        {
            vFloat c = coeffs[10];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 11) {
        {
            vFloat c = coeffs[11];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[10];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 12) {
        {
            vFloat c = coeffs[12];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[11];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[10];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 13) {
        {
            vFloat c = coeffs[13];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[12];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[11];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[10];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 14) {
        {
            vFloat c = coeffs[14];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[13];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[12];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[11];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[10];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 15) {
        {
            vFloat c = coeffs[15];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[14];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[13];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[12];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[11];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[10];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    } else if constexpr (DEGREE == 16) {
        {
            vFloat c = coeffs[16];
            result1 = c;
            result2 = c;
        }
        {
            vFloat c = coeffs[15];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[14];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[13];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[12];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[11];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[10];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[9];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[8];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[7];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[6];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[5];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[4];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[3];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[2];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[1];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
        {
            vFloat c = coeffs[0];
            result1 = result1 * x1 + c;
            result2 = result2 * x2 + c;
        }
    }
}

// Dual-eval x²-Horner for polynomials with parity structure.
// Two interleaved chains in x² basis for ILP, same stride-2 pattern.
#if defined(POLY_PARITY_ODD)
template <uint32_t DEGREE>
inline void eval_polynomial_dual_parity(
    const float* coeffs, vFloat x1, vFloat x2, vFloat x1_sq, vFloat x2_sq, vFloat& result1, vFloat& result2) {
    constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = (TOP - 1) / 2;
    {
        vFloat c = coeffs[TOP];
        result1 = c;
        result2 = c;
    }
#pragma GCC unroll 16
    for (int k = 1; k <= STEPS; k++) {
        vFloat c = coeffs[TOP - 2 * k];
        result1 = result1 * x1_sq + c;
        result2 = result2 * x2_sq + c;
    }
    result1 = result1 * x1;
    result2 = result2 * x2;
}
#elif defined(POLY_PARITY_EVEN)
template <uint32_t DEGREE>
inline void eval_polynomial_dual_parity(
    const float* coeffs, vFloat x1, vFloat x2, vFloat x1_sq, vFloat x2_sq, vFloat& result1, vFloat& result2) {
    constexpr int TOP = (DEGREE % 2 == 0) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = TOP / 2;
    {
        vFloat c = coeffs[TOP];
        result1 = c;
        result2 = c;
    }
#pragma GCC unroll 16
    for (int k = 1; k <= STEPS; k++) {
        vFloat c = coeffs[TOP - 2 * k];
        result1 = result1 * x1_sq + c;
        result2 = result2 * x2_sq + c;
    }
}
#endif

#endif  // USE_DUAL_EVAL

// Generic piecewise polynomial LUT for any degree/depth combination
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;  // Skip boundary points

    // Compile-time validation
    static_assert(
        LUT_SIZE == (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * COEFFS_PER_SEGMENT),
        "LUT_SIZE must equal (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)");

    // Process all 32 destination registers
    for (int d = 0; d < 32; d++) {
        vFloat x_orig = dst_reg[d];

#if defined(RANGE_REDUCTION_EXP)
        // Exp range reduction: clamp extreme values, then reduce to [-ln(2)/2, ln(2)/2]
        constexpr float EXP_OVERFLOW = 88.5f;
        constexpr float EXP_UNDERFLOW = -88.5f;
        vFloat x;
        vInt k_int;
        exp_reduce(x_orig, x, k_int);
#elif defined(RANGE_REDUCTION_TRIG)
        // Trig range reduction: reduce to [-π/2, π/2]
        vFloat x;
        vInt q_int;
        trig_reduce(x_orig, x, q_int);
#elif defined(RANGE_REDUCTION_TAN)
        // Tan range reduction: reduce to [-π/4, π/4] (Cody-Waite)
        vFloat x;
        vInt j_int;
        tan_reduce(x_orig, x, j_int);
#elif defined(RANGE_REDUCTION_LOG)
        // Log range reduction: extract mantissa m ∈ [1, 2) and exponent e
        // log(x) = e*ln(2) + log(m)
        vFloat x;
        vInt e_int;
        log_reduce(x_orig, x, e_int);
#elif defined(RANGE_REDUCTION_CBRT)
        // Extract BIASED exponent (always non-negative) and sign BEFORE poly eval.
        // Use exexp_nodebias to avoid SFPU's sign-magnitude format for negative debiased values.
        vInt cbrt_biased_e = exexp_nodebias(setsgn(x_orig, 0));
        vInt cbrt_sign = reinterpret<vInt>(x_orig) & 0x80000000;
        vFloat x = cbrt_reduce_m(x_orig);
#else
        vFloat x = x_orig;
#endif

#if defined(BASIS_INPUT_ABS_X)
        vFloat x_eval = setsgn(x_orig, 0);
        vFloat& x_clamped = x_eval;
#else
        vFloat& x_eval = x;
        // Clamping unnecessary: segment cascade v_if(x >= boundary) naturally selects
        // the edge segment for out-of-range inputs. Removing saves SFPU registers.
        vFloat& x_clamped = x;
#endif

        // Start with segment 0 coefficients
        vFloat result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET], x_eval);

        // Update with correct segment using plain for-loop
        // Works correctly on all architectures without kernel bloat
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            v_if(x_clamped >= lut[seg]) {
                result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT], x_eval);
            }
            v_endif;
        }

#if defined(BASIS_MUL_ABS_X_BEFORE_POST)
        result = result * x_eval;
#endif

#if defined(BASIS_AFFINE_EVEN)
        result = BASIS_AFFINE_BIAS + BASIS_AFFINE_SCALE * x_orig + (BASIS_AFFINE_EVEN_SCALE * x_eval) * result;
#endif

#if defined(BASIS_CLAMP_MAX)
        vFloat basis_clamp_max_value = BASIS_CLAMP_MAX_VALUE;
        vec_min_max(result, basis_clamp_max_value);
#endif

#if defined(BASIS_POST_SIGN_X)
        result = copysgn(result, x_orig);
#endif

#if defined(BASIS_LEFT_TAIL_ZERO)
        v_if(x_orig < BASIS_LEFT_TAIL_ZERO_THRESHOLD) { result = 0.0f; }
        v_endif;
#endif

#if defined(BASIS_RIGHT_TAIL_IDENTITY)
        v_if(x_orig > BASIS_RIGHT_TAIL_IDENTITY_THRESHOLD) { result = x_orig; }
        v_endif;
#endif

#if defined(RANGE_REDUCTION_EXP)
        // Expand: result = poly_result * 2^k, with overflow/underflow clamping
        v_if(x_orig > EXP_OVERFLOW) { result = std::numeric_limits<float>::infinity(); }
        v_elseif(x_orig < EXP_UNDERFLOW) { result = 0.0f; }
        v_else { result = exp_expand(result, k_int); }
        v_endif;
#elif defined(RANGE_REDUCTION_TRIG)
        // Trig expand: negate result if quotient is odd
        result = trig_expand(result, q_int);
#elif defined(RANGE_REDUCTION_TAN)
        // Tan expand: j even → poly(a), j odd → -1/poly(a)
        result = tan_expand(result, j_int);
#elif defined(RANGE_REDUCTION_LOG)
        // Log expand: log(x) = e*ln(2) + log(m)
        // Handle special cases: log(0) = -inf, log(negative) = NaN
        v_if(x_orig < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(x_orig == 0.0f) { result = -std::numeric_limits<float>::infinity(); }
        v_else { result = log_expand(result, e_int); }
        v_endif;
#elif defined(RANGE_REDUCTION_CBRT)
        // Compute q = floor(e/3) and r = e mod 3 using biased exponent.
        // All arithmetic stays in float to avoid SFPU's sign-magnitude int format.
        {
            constexpr float ONE_THIRD_C = 0.3333333333333333f;
            const vFloat magic = ckernel::sfpu::Converter::as_float(0x4B400000U);

            // Convert biased exponent to float and debias: e_float = biased - 127
            vFloat e_float = int32_to_float(cbrt_biased_e, RoundMode::Nearest) - 127.0f;

            // q_float ≈ e/3, then round to nearest integer
            vFloat q_approx = e_float * ONE_THIRD_C;
            vFloat q_rounded = q_approx + magic;
            vInt q = reinterpret<vInt>(q_rounded) - reinterpret<vInt>(magic);

            // Get q as float WITHOUT int32_to_float (avoids sign-magnitude)
            vFloat q_back = q_rounded - magic;

            // r = e - 3*q in float, then convert to int via magic number
            vFloat r_float = e_float - (q_back + q_back + q_back);
            vInt r = reinterpret<vInt>(r_float + magic) - reinterpret<vInt>(magic);

            // Corrections for r ∈ {0,1,2}
            v_if(r < 0) {
                q = q - 1;
                r = r + 3;
            }
            v_endif;
            v_if(r > 2) {
                q = q + 1;
                r = r - 3;
            }
            v_endif;

            // Handle x=0 (biased_exp = 0 for zero/denorm)
            v_if(cbrt_biased_e < 1) { result = 0.0f; }
            v_else { result = cbrt_expand(result, q, r, cbrt_sign); }
            v_endif;
        }
#endif

#ifdef HAS_CRITICAL_POINT
        // Override with exact value at critical point for perfect accuracy
        // (e.g., cosh(0)=1, tanh(0)=0, sigmoid(0)=0.5)
        v_if(x == lut[CRITICAL_IDX]) { result = CRITICAL_VALUE; }
        v_endif;
#endif

        result = apply_output_postcompose(result);

        // bf16 dst: RNE-round before the store. SFPSTORE narrows fp32->bf16 by
        // truncation (RTZ) in hardware; rounding here (sfpstochrnd RND_EVEN)
        // makes the already-bf16 value lossless under SFPSTORE and recovers the
        // half-ULP that RTZ would otherwise drop (the ML-pass output bias).
#ifdef USE_BF16
        result = convert<vFloat16b>(result, RoundMode::Nearest);
#endif
        dst_reg[d] = result;
    }
}

#ifdef USE_DUAL_EVAL
// Dual-evaluation version of piecewise_generic_lut.
// Processes two destination registers per iteration (d and d+1), exploiting
// SFPU instruction-level parallelism by interleaving two independent Horner chains.
//
// Segment 0 uses eval_polynomial_dual for full ILP benefit.
// Segments 1..N-1 use independent single-eval (x1 and x2 may be in different segments).
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dual(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    static_assert(
        LUT_SIZE == (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * COEFFS_PER_SEGMENT),
        "LUT_SIZE must equal (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)");

    // Process pairs of destination registers (32 total → 16 iterations)
    for (int d = 0; d < 32; d += 2) {
        vFloat x_orig1 = dst_reg[d];
        vFloat x_orig2 = dst_reg[d + 1];

#if defined(RANGE_REDUCTION_EXP)
        constexpr float EXP_OVERFLOW = 88.5f;
        constexpr float EXP_UNDERFLOW = -88.5f;
        vFloat x1, x2;
        vInt k_int1, k_int2;
        exp_reduce(x_orig1, x1, k_int1);
        exp_reduce(x_orig2, x2, k_int2);
#elif defined(RANGE_REDUCTION_TRIG)
        vFloat x1, x2;
        vInt q_int1, q_int2;
        trig_reduce(x_orig1, x1, q_int1);
        trig_reduce(x_orig2, x2, q_int2);
#elif defined(RANGE_REDUCTION_TAN)
        vFloat x1, x2;
        vInt j_int1, j_int2;
        tan_reduce(x_orig1, x1, j_int1);
        tan_reduce(x_orig2, x2, j_int2);
#elif defined(RANGE_REDUCTION_LOG)
        vFloat x1, x2;
        vInt e_int1, e_int2;
        log_reduce(x_orig1, x1, e_int1);
        log_reduce(x_orig2, x2, e_int2);
#elif defined(RANGE_REDUCTION_CBRT)
        // Extract BIASED exponent and sign BEFORE poly eval
        vInt cbrt_biased_e1 = exexp_nodebias(setsgn(x_orig1, 0));
        vInt cbrt_sign1 = reinterpret<vInt>(x_orig1) & 0x80000000;
        vInt cbrt_biased_e2 = exexp_nodebias(setsgn(x_orig2, 0));
        vInt cbrt_sign2 = reinterpret<vInt>(x_orig2) & 0x80000000;
        vFloat x1 = cbrt_reduce_m(x_orig1);
        vFloat x2 = cbrt_reduce_m(x_orig2);
#else
        vFloat x1 = x_orig1;
        vFloat x2 = x_orig2;
#endif

#if defined(BASIS_INPUT_ABS_X)
        x1 = setsgn(x_orig1, 0);
        x2 = setsgn(x_orig2, 0);
#endif

        // Clamping unnecessary: segment cascade v_if(x >= boundary) naturally selects
        // the edge segment for out-of-range inputs. Removing saves SFPU registers.
        vFloat& x1_clamped = x1;
        vFloat& x2_clamped = x2;

        // Segment 0: dual evaluation exploits ILP for interleaved Horner chains
        vFloat result1, result2;
        eval_polynomial_dual<POLY_DEGREE>(&lut[COEFF_OFFSET], x1, x2, result1, result2);

        // Segments 1..N-1: independent selection (x1 and x2 may be in different segments)
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            v_if(x1_clamped >= lut[seg]) {
                result1 = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT], x1);
            }
            v_endif;
            v_if(x2_clamped >= lut[seg]) {
                result2 = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT], x2);
            }
            v_endif;
        }

#if defined(BASIS_MUL_ABS_X_BEFORE_POST)
        result1 = result1 * x1;
        result2 = result2 * x2;
#endif

#if defined(BASIS_AFFINE_EVEN)
        result1 = BASIS_AFFINE_BIAS + BASIS_AFFINE_SCALE * x_orig1 + (BASIS_AFFINE_EVEN_SCALE * x1) * result1;
        result2 = BASIS_AFFINE_BIAS + BASIS_AFFINE_SCALE * x_orig2 + (BASIS_AFFINE_EVEN_SCALE * x2) * result2;
#endif

#if defined(BASIS_CLAMP_MAX)
        vFloat basis_clamp_max_value1 = BASIS_CLAMP_MAX_VALUE;
        vFloat basis_clamp_max_value2 = BASIS_CLAMP_MAX_VALUE;
        vec_min_max(result1, basis_clamp_max_value1);
        vec_min_max(result2, basis_clamp_max_value2);
#endif

#if defined(BASIS_POST_SIGN_X)
        result1 = copysgn(result1, x_orig1);
        result2 = copysgn(result2, x_orig2);
#endif

#if defined(BASIS_LEFT_TAIL_ZERO)
        v_if(x_orig1 < BASIS_LEFT_TAIL_ZERO_THRESHOLD) { result1 = 0.0f; }
        v_endif;
        v_if(x_orig2 < BASIS_LEFT_TAIL_ZERO_THRESHOLD) { result2 = 0.0f; }
        v_endif;
#endif

#if defined(BASIS_RIGHT_TAIL_IDENTITY)
        v_if(x_orig1 > BASIS_RIGHT_TAIL_IDENTITY_THRESHOLD) { result1 = x_orig1; }
        v_endif;
        v_if(x_orig2 > BASIS_RIGHT_TAIL_IDENTITY_THRESHOLD) { result2 = x_orig2; }
        v_endif;
#endif

#if defined(RANGE_REDUCTION_EXP)
        v_if(x_orig1 > EXP_OVERFLOW) { result1 = std::numeric_limits<float>::infinity(); }
        v_elseif(x_orig1 < EXP_UNDERFLOW) { result1 = 0.0f; }
        v_else { result1 = exp_expand(result1, k_int1); }
        v_endif;
        v_if(x_orig2 > EXP_OVERFLOW) { result2 = std::numeric_limits<float>::infinity(); }
        v_elseif(x_orig2 < EXP_UNDERFLOW) { result2 = 0.0f; }
        v_else { result2 = exp_expand(result2, k_int2); }
        v_endif;
#elif defined(RANGE_REDUCTION_TRIG)
        result1 = trig_expand(result1, q_int1);
        result2 = trig_expand(result2, q_int2);
#elif defined(RANGE_REDUCTION_TAN)
        result1 = tan_expand(result1, j_int1);
        result2 = tan_expand(result2, j_int2);
#elif defined(RANGE_REDUCTION_LOG)
        // Log expand: handle special cases and reconstruct log(x) = e*ln(2) + log(m)
        v_if(x_orig1 < 0.0f) { result1 = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(x_orig1 == 0.0f) { result1 = -std::numeric_limits<float>::infinity(); }
        v_else { result1 = log_expand(result1, e_int1); }
        v_endif;
        v_if(x_orig2 < 0.0f) { result2 = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(x_orig2 == 0.0f) { result2 = -std::numeric_limits<float>::infinity(); }
        v_else { result2 = log_expand(result2, e_int2); }
        v_endif;
#elif defined(RANGE_REDUCTION_CBRT)
        // Cbrt expand using biased exponent — all arithmetic in float
        {
            constexpr float ONE_THIRD_C = 0.3333333333333333f;
            const vFloat magic = ckernel::sfpu::Converter::as_float(0x4B400000U);

            // Result 1
            vFloat e_float1 = int32_to_float(cbrt_biased_e1, RoundMode::Nearest) - 127.0f;
            vFloat q_approx1 = e_float1 * ONE_THIRD_C;
            vFloat q_rounded1 = q_approx1 + magic;
            vInt q1 = reinterpret<vInt>(q_rounded1) - reinterpret<vInt>(magic);
            vFloat q_back1 = q_rounded1 - magic;
            vFloat r_float1 = e_float1 - (q_back1 + q_back1 + q_back1);
            vInt r1 = reinterpret<vInt>(r_float1 + magic) - reinterpret<vInt>(magic);
            v_if(r1 < 0) {
                q1 = q1 - 1;
                r1 = r1 + 3;
            }
            v_endif;
            v_if(r1 > 2) {
                q1 = q1 + 1;
                r1 = r1 - 3;
            }
            v_endif;
            v_if(cbrt_biased_e1 < 1) { result1 = 0.0f; }
            v_else { result1 = cbrt_expand(result1, q1, r1, cbrt_sign1); }
            v_endif;

            // Result 2
            vFloat e_float2 = int32_to_float(cbrt_biased_e2, RoundMode::Nearest) - 127.0f;
            vFloat q_approx2 = e_float2 * ONE_THIRD_C;
            vFloat q_rounded2 = q_approx2 + magic;
            vInt q2 = reinterpret<vInt>(q_rounded2) - reinterpret<vInt>(magic);
            vFloat q_back2 = q_rounded2 - magic;
            vFloat r_float2 = e_float2 - (q_back2 + q_back2 + q_back2);
            vInt r2 = reinterpret<vInt>(r_float2 + magic) - reinterpret<vInt>(magic);
            v_if(r2 < 0) {
                q2 = q2 - 1;
                r2 = r2 + 3;
            }
            v_endif;
            v_if(r2 > 2) {
                q2 = q2 + 1;
                r2 = r2 - 3;
            }
            v_endif;
            v_if(cbrt_biased_e2 < 1) { result2 = 0.0f; }
            v_else { result2 = cbrt_expand(result2, q2, r2, cbrt_sign2); }
            v_endif;
        }
#endif

#ifdef HAS_CRITICAL_POINT
        v_if(x1 == lut[CRITICAL_IDX]) { result1 = CRITICAL_VALUE; }
        v_endif;
        v_if(x2 == lut[CRITICAL_IDX]) { result2 = CRITICAL_VALUE; }
        v_endif;
#endif

        result1 = apply_output_postcompose(result1);
        result2 = apply_output_postcompose(result2);

        // bf16 dst: RNE-round before the store (see single-eval note above).
#ifdef USE_BF16
        result1 = convert<vFloat16b>(result1, RoundMode::Nearest);
        result2 = convert<vFloat16b>(result2, RoundMode::Nearest);
#endif
        dst_reg[d] = result1;
        dst_reg[d + 1] = result2;
    }
}
#endif  // USE_DUAL_EVAL

#if EVAL_METHOD_IS_STANDALONE
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dispatch(const std::array<float, LUT_SIZE>& lut) {
#ifdef USE_DUAL_EVAL
    piecewise_generic_lut_dual<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#else
    piecewise_generic_lut<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#endif
}
#endif

// Include specialized implementations for common segment counts
// These use manual unrolling to work around Wormhole SFPU compiler bug
#if !EVAL_METHOD_IS_STANDALONE
#include "piecewise_generic_specialized.cpp"
#endif

#if defined(AFFINE_COLLAPSE) && !defined(AFFINE_IDENTITY)
// Affine collapse: the whole fit is y = c0 + c1*x. One SFPMAD per element,
// bypassing the entire segment cascade / LUT machinery. c0/c1 are SFPMAD
// immediates (single hoistable load each), so the per-element body is one MAD.
inline void affine_collapse_eval() {
    constexpr float C0 = AFFINE_C0;
    constexpr float C1 = AFFINE_C1;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat y = C1 * x + C0;
        y = apply_output_postcompose(y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

#if defined(CLAMPED_AFFINE_COLLAPSE)
// Clamped affine collapse: y = min(max(c0 + c1*x, low), high). This is the
// exact algebraic form of piecewise-linear CSVs with flat outer regions, but is
// driven only by coefficient pattern detection in run_csv.sh.
inline void clamped_affine_collapse_eval() {
    constexpr float C0 = CLAMPED_AFFINE_C0;
    constexpr float C1 = CLAMPED_AFFINE_C1;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat y = C1 * x + C0;
#ifdef CLAMPED_AFFINE_HAS_MIN
        vFloat lo = CLAMPED_AFFINE_MIN;
        vec_min_max(lo, y);  // y = max(lo, y)
#endif
#ifdef CLAMPED_AFFINE_HAS_MAX
        vFloat hi = CLAMPED_AFFINE_MAX;
        vec_min_max(y, hi);  // y = min(y, hi)
#endif
        y = apply_output_postcompose(y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

#if defined(ABS_VALUE)
// y=abs(x). Exact whole-function lowering for two affine pieces (-x, +x).
inline void abs_value_eval() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat y = setsgn(x, 0);
        y = apply_output_postcompose(y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

#if defined(THRESHOLD_IDENTITY_SELECT)
// y=x for |x|>lambda, else zero. Equality belongs to zero.
inline void threshold_identity_select_eval() {
    constexpr float LAMBDA = THRESHOLD_IDENTITY_LAMBDA;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat y = x;
        vFloat ax = setsgn(x, 0);
        v_if(ax <= LAMBDA) { y = 0.0f; }
        v_endif;
        y = apply_output_postcompose(y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

#if defined(THRESHOLD_SOFTSHIFT_SELECT)
// y=sign(x)*(abs(x)-lambda) for |x|>lambda, else zero. Equality belongs to zero.
inline void threshold_softshift_select_eval() {
    constexpr float LAMBDA = THRESHOLD_SOFTSHIFT_LAMBDA;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat ax = setsgn(x, 0);
        vFloat y = 0.0f;
        v_if(ax > LAMBDA) {
            y = ax - LAMBDA;
            v_if(x < 0.0f) { y = -y; }
            v_endif;
        }
        v_endif;
        y = apply_output_postcompose(y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

#if defined(GATED_AFFINE_PRODUCT) || defined(GATED_QUADRATIC_COLLAPSE)
// y = x * clamp(q0 + q1*x, 0, 1).
inline void gated_affine_product_eval() {
    constexpr float Q0 = GATED_QUADRATIC_Q0;
    constexpr float Q1 = GATED_QUADRATIC_Q1;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat gate = Q1 * x + Q0;
        vFloat zero = 0.0f;
        vFloat one = 1.0f;
        vec_min_max(zero, gate);  // gate = max(0, gate)
        vec_min_max(gate, one);   // gate = min(gate, 1)
        vFloat y = x * gate;
        y = apply_output_postcompose(y);
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[d] = y;
    }
}
#endif

}  // namespace sfpi
#endif

void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

#ifdef EMBEDDED_LUT
    // Embedded LUT mode: LUT is compiled directly into the kernel
    // Header must define: POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE, LUT_DATA
    // This provides zero L1 memory overhead for the LUT
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Use embedded LUT data directly from header
    constexpr uint32_t lut_size = LUT_SIZE;
    constexpr uint32_t poly_degree = POLY_DEGREE;
    constexpr uint32_t num_segments = NUM_SEGMENTS;
    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;
#else
    // Generic LUT mode: LUT is loaded from L1 circular buffer
    // This allows runtime LUT generation and sharing across cores
    [[maybe_unused]] float input_min = get_arg_val<float>(1);
    [[maybe_unused]] float input_max = get_arg_val<float>(2);
    constexpr uint32_t lut_size = get_compile_time_arg_val(0);
    constexpr uint32_t poly_degree = get_compile_time_arg_val(1);
    constexpr uint32_t num_segments = get_compile_time_arg_val(2);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_lut = tt::CBIndex::c_25;

    // Get LUT array from L1 memory as float array directly
    using lut_t = std::array<float, lut_size>;
    auto p_lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);
#endif

    init_sfpu(cb_in, cb_out);

#if defined(HW_PRELOAD) && defined(TRISC_MATH)
    // GENERIC constant-pool preload: program the 3 hottest loop-invariant
    // constants for this exponent-ALU kind into the programmable const registers
    // ONCE (they persist across every replayed body and every tile). The
    // per-kind ranking matches the *_hw_eval_preloaded readers above.
#if defined(EXPONENT_ALU_EXP2)
#if !defined(EXP_HW_COMPOSE_SIGMOID) && !defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)
    // Under sigmoid compose, prgm0 is reserved for sfpu_reciprocal (set to 2.0 by
    // sfpu_reciprocal_init); MULT is hoisted into an LREG instead (see hw_reduce).
    sfpi::vConstFloatPrgm0 = EXP_HW_MULT;  // touched every element
#endif
#if defined(EXP_HW_FUSED)
    // FUSED: prgm1/prgm2 carry the pre-scaled coeffs c[k]*2^-23k so the
    // per-element Horner runs over the UNNORMALIZED exman fraction (no
    // `* 0x1p-23f`). Scale is a compile-time constant per coeff index.
    sfpi::vConstFloatPrgm1 = EXP_HW_COEFFS[EXP_HW_DEGREE] * sfpi::exp_hw_fused_scale(EXP_HW_DEGREE);
    sfpi::vConstFloatPrgm2 =
        (EXP_HW_DEGREE >= 1) ? EXP_HW_COEFFS[EXP_HW_DEGREE - 1] * sfpi::exp_hw_fused_scale(EXP_HW_DEGREE - 1) : 0.0f;
#else
    sfpi::vConstFloatPrgm1 = EXP_HW_COEFFS[EXP_HW_DEGREE];
    sfpi::vConstFloatPrgm2 = (EXP_HW_DEGREE >= 1) ? EXP_HW_COEFFS[EXP_HW_DEGREE - 1] : 0.0f;
#endif
#elif defined(EXPONENT_ALU_LOG2)
    sfpi::vConstFloatPrgm0 = LOG_HW_SCALE;
    sfpi::vConstFloatPrgm1 = LOG_HW_COEFFS[LOG_HW_DEGREE];
    sfpi::vConstFloatPrgm2 = (LOG_HW_DEGREE >= 1) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 1] : 0.0f;
#elif defined(EXPONENT_ALU_POW)
    // Only plain sqrt (ROOT_N==2, no reciprocal) uses the preload fast path;
    // cbrt/rsqrt fall back to the general evaluator (see pow_hw_eval_preloaded),
    // which reads no prgm registers, so we must not clobber them there.
#if (POW_HW_ROOT_N == 2) && !defined(POW_HW_RECIPROCAL)
    sfpi::vConstFloatPrgm0 = POW_HW_SCALE_C1;  // sqrt(2)
    sfpi::vConstFloatPrgm1 = POW_HW_COEFFS[POW_HW_DEGREE];
    sfpi::vConstFloatPrgm2 = (POW_HW_DEGREE >= 1) ? POW_HW_COEFFS[POW_HW_DEGREE - 1] : 0.0f;
#endif
#endif
#endif

#if defined(EVAL_METHOD_NEWTON_ROOT) && defined(TRISC_MATH)
    // Newton-Raphson magic-seed root: preload the seed magic + Newton coeffs into
    // the programmable const registers ONCE so the recorded per-element body
    // reloads none of them (mirrors native sqrt_init).
    //   sqrt : prgm0=magic, prgm1=C1, prgm2=C2 (SQRT_23-bit double-Newton).
    //   rsqrt: prgm0=inverse-sqrt magic, prgm1=C1 (=1.5 step constant).
    //   cbrt : native-style magic-root uses prgm0/1/2 as correction coeffs.
#if (NEWTON_ROOT_N == 3) && defined(NEWTON_ROOT_ALGORITHM_CBRT_MAGIC)
    sfpi::vConstFloatPrgm0 = NEWTON_ROOT_C0;
    sfpi::vConstFloatPrgm1 = NEWTON_ROOT_C1;
    sfpi::vConstFloatPrgm2 = NEWTON_ROOT_C2;
#elif (NEWTON_ROOT_N != 3)
    sfpi::vConstIntPrgm0 = NEWTON_ROOT_MAGIC;
    sfpi::vConstFloatPrgm1 = NEWTON_ROOT_C1;  // sqrt: C1; rsqrt: 1.5 step constant
    sfpi::vConstFloatPrgm2 = NEWTON_ROOT_C2;  // sqrt: C2; rsqrt: unused
#endif
#endif

#if defined(EVAL_METHOD_TRIG_RESIDUAL) && defined(TRISC_MATH)
    // Four-part Cody-Waite tail constants and 1/PI multiplier for the standalone
    // trig residual evaluator.
#if defined(TRIG_RESIDUAL_PHASE_SINE_PI_ODD)
    sfpi::vConstFloatPrgm0 = -0x1.51p-21f;
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-33f;
#else
    sfpi::vConstFloatPrgm0 = -0x1.51p-22f;
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-34f;
#endif
    sfpi::vConstFloatPrgm2 = 0.31830987334251404f;
#endif

#if defined(EVAL_METHOD_TAN_STANDALONE) && defined(TRISC_MATH)
    sfpi::vConstFloatPrgm0 = -0x1.51p-22f;
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-34f;
    sfpi::vConstFloatPrgm2 = 0.63661974668502808f;
#endif

#if defined(RANGE_REDUCTION_TAN) && defined(TRISC_MATH)
    ckernel::sfpu::sfpu_reciprocal_init<false>();
#endif
#if (defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT)) && defined(TRISC_MATH)
    ckernel::sfpu::sfpu_reciprocal_init<false>();
#endif

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);

// All degree parameters are constexpr — use them directly as template args.
// No dispatch table needed; works for ANY poly_degree automatically.
#ifdef TRISC_MATH
#if TT_ACT_EVAL_KIND == TT_ACT_EVAL_IDENTITY || defined(AFFINE_IDENTITY)
        // Identity collapse (y = x): copy_tile already placed x in dst; the SFPU
        // eval is a pure no-op, so skip it entirely (no Horner, no LUT, no cascade).
        (void)p_lut;
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_AFFINE || defined(AFFINE_COLLAPSE)
        // Affine collapse (y = c0 + c1*x): one SFPMAD per element, no cascade.
        (void)p_lut;
        sfpi::affine_collapse_eval();
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_CLAMPED_AFFINE || defined(CLAMPED_AFFINE_COLLAPSE)
        // Clamped affine collapse: one SFPMAD plus optional min/max clamps.
        (void)p_lut;
        sfpi::clamped_affine_collapse_eval();
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_ABS_VALUE || defined(ABS_VALUE)
        (void)p_lut;
        sfpi::abs_value_eval();
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_THRESHOLD_IDENTITY || defined(THRESHOLD_IDENTITY_SELECT)
        (void)p_lut;
        sfpi::threshold_identity_select_eval();
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_THRESHOLD_SOFTSHIFT || defined(THRESHOLD_SOFTSHIFT_SELECT)
        (void)p_lut;
        sfpi::threshold_softshift_select_eval();
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_GATED_AFFINE_PRODUCT || defined(GATED_AFFINE_PRODUCT) || defined(GATED_QUADRATIC_COLLAPSE)
        (void)p_lut;
        sfpi::gated_affine_product_eval();
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_EXPONENT_ALU || defined(EVAL_METHOD_EXPONENT_ALU)
        (void)p_lut;
        for (int d = 0; d < 32; d++) {
#if defined(EXPONENT_ALU_EXP2)
            sfpi::vFloat y = sfpi::exp_hw_eval<EXP_HW_DEGREE>(sfpi::dst_reg[d]);
#elif defined(EXPONENT_ALU_LOG2)
            sfpi::vFloat y = sfpi::log_hw_eval<LOG_HW_DEGREE>(sfpi::dst_reg[d]);
#elif defined(EXPONENT_ALU_POW)
            sfpi::vFloat y = sfpi::pow_hw_eval<POW_HW_DEGREE>(sfpi::dst_reg[d]);
#else
#error "EVAL_METHOD_EXPONENT_ALU requires EXPONENT_ALU_EXP2/LOG2/POW"
#endif
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_NEWTON_ROOT || defined(EVAL_METHOD_NEWTON_ROOT)
        (void)p_lut;
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat y = sfpi::newton_root_eval<0>(sfpi::dst_reg[d]);
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_TRIG_RESIDUAL || defined(EVAL_METHOD_TRIG_RESIDUAL)
        (void)p_lut;
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat y = sfpi::trig_residual_odd_eval<TRIG_RESIDUAL_DEGREE>(sfpi::dst_reg[d]);
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_TAN_STANDALONE || defined(EVAL_METHOD_TAN_STANDALONE)
        (void)p_lut;
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat y = sfpi::tan_standalone_eval<TAN_STANDALONE_DEGREE>(sfpi::dst_reg[d]);
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_ASIN_ACOS || defined(EVAL_METHOD_ASIN_ACOS)
        (void)p_lut;
        for (int d = 0; d < 32; d++) {
            sfpi::vFloat y = sfpi::asin_acos_eval(sfpi::dst_reg[d]);
#ifdef USE_BF16
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
#endif
            sfpi::dst_reg[d] = y;
        }
#else
        sfpi::piecewise_generic_lut_dispatch<poly_degree, num_segments, lut_size>(*p_lut);
#endif
#endif

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }
}

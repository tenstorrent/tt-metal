// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// ======================================================================
// Softplus via abs(x) symmetry + residual function
//
// Uses the identity: softplus(-x) = softplus(x) - x
// Defining f(a) = ln(1 + exp(-a)) for a >= 0:
//   softplus(t) = t + f(t)   for t >= 0
//   softplus(t) = f(-t)      for t < 0
//
// FP32: degree-8 polynomial for f(a) on [0, 5] + inline exp + 3-term Taylor tail
// BF16: degree-6 polynomial (bf16-accurate, <0.28 ULP) + tail clamped to 0
//       (residual < exp(-5) = 0.0067 for a > 5, below bf16 rounding vs the t>0 term,
//        so the expensive exp tail is unnecessary at bf16 precision)
// ======================================================================

constexpr float SOFTPLUS_POLY_BOUNDARY = 5.0f;

// FP32 residual polynomial: f(a) = ln(1+exp(-a)) on [0, 5], degree 8
constexpr float SOFTPLUS_POLY_C0 = 6.9310557842e-01f;
constexpr float SOFTPLUS_POLY_C1 = -4.9926245213e-01f;
constexpr float SOFTPLUS_POLY_C2 = 1.2186349183e-01f;
constexpr float SOFTPLUS_POLY_C3 = 5.6753782555e-03f;
constexpr float SOFTPLUS_POLY_C4 = -1.0528374463e-02f;
constexpr float SOFTPLUS_POLY_C5 = 2.7290175203e-03f;
constexpr float SOFTPLUS_POLY_C6 = -3.4358495031e-04f;
constexpr float SOFTPLUS_POLY_C7 = 2.1285692128e-05f;
constexpr float SOFTPLUS_POLY_C8 = -4.8245715334e-07f;

// BF16 residual polynomial: f(a) = ln(1+exp(-a)) on [0, 5], degree 6
// (ULP-weighted minimax fit; max error < 0.28 bf16 ULP over the domain)
constexpr float SOFTPLUS_BF16_POLY_C0 = 6.9423984729e-01f;
constexpr float SOFTPLUS_BF16_POLY_C1 = -5.0932420424e-01f;
constexpr float SOFTPLUS_BF16_POLY_C2 = 1.4279095486e-01f;
constexpr float SOFTPLUS_BF16_POLY_C3 = -1.3000584069e-02f;
constexpr float SOFTPLUS_BF16_POLY_C4 = -1.8627923291e-03f;
constexpr float SOFTPLUS_BF16_POLY_C5 = 5.0152968088e-04f;
constexpr float SOFTPLUS_BF16_POLY_C6 = -3.1273466851e-05f;

// ======================================================================
// Lightweight inline exp(x) for negative x (tail region).
// Adapted from gelu's x_times_exp_negative_tail (ckernel_sfpu_gelu.h).
// Uses Cody-Waite range reduction + Taylor polynomial.
// BF16: degree 5 (~15 ops), FP32: degree 7 (~19 ops).
// ======================================================================
sfpi_inline sfpi::vFloat softplus_exp_negative(sfpi::vFloat x) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    constexpr float LN2_HI = -0.6931152343750000f;
    constexpr float LN2_LO = -3.19461832987e-05f;

    // Range reduction: x = k*ln(2) + r
    sfpi::vFloat z = x * INV_LN2;
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

    // Cody-Waite: r = x - k*ln(2) in extended precision
    sfpi::vFloat r = k * LN2_HI + x;
    r = k * LN2_LO + r;

    // exp(r) via Taylor polynomial, |r| < 0.5
#ifdef INP_FLOAT32
    // FP32: degree 7 for < 1 ULP
    sfpi::vFloat poly = PolynomialEvaluator::eval(
        r, 1.0f, 1.0f, 0.5f, 0.166666667f, 0.0416666667f, 0.00833333333f, 0.00138888889f, 0.000198412698f);
#else
    // BF16: degree 5 sufficient
    sfpi::vFloat poly = PolynomialEvaluator::eval(r, 1.0f, 1.0f, 0.5f, 0.166666667f, 0.0416666667f, 0.00833333333f);
#endif

    // Scale by 2^k via exponent manipulation
    sfpi::vInt p_exp = sfpi::exexp(poly, sfpi::ExponentMode::Biased);
    sfpi::vInt new_exp = p_exp + k_int;

    // FTZ: if exponent underflows, result is 0
    sfpi::vFloat result = 0.0f;
    v_if(new_exp > 0) { result = sfpi::setexp(poly, new_exp); }
    v_endif;

    return result;
}

// The three lowest-order residual-polynomial coefficients live in the SFPU's programmable
// constant registers. SFPMAD names a CREG directly in an operand field, so each one costs
// zero instructions per element instead of the two SFPLOADI a full-fp32 literal needs --
// and the value keeps its exact fp32 bit pattern, so this is bit-exact, not an accuracy
// trade. Nothing else in softplus's call graph touches Prgm0/1/2 (its exp tail uses a
// local 2^23+2^22 constant, which is bf16-exact and already free).
//
// Reached from _llk_math_eltwise_unary_sfpu_init_'s SfpuType::softplus arm, which both
// production (SFPU_UNARY_INIT) and the tt-llk harness run before the calculate step.
#ifdef INP_FLOAT32
#define SOFTPLUS_CREG_C0 SOFTPLUS_POLY_C0
#define SOFTPLUS_CREG_C1 SOFTPLUS_POLY_C1
#define SOFTPLUS_CREG_C2 SOFTPLUS_POLY_C2
#else
#define SOFTPLUS_CREG_C0 SOFTPLUS_BF16_POLY_C0
#define SOFTPLUS_CREG_C1 SOFTPLUS_BF16_POLY_C1
#define SOFTPLUS_CREG_C2 SOFTPLUS_BF16_POLY_C2
#endif

inline void softplus_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = SOFTPLUS_CREG_C0;
    sfpi::vConstFloatPrgm1 = SOFTPLUS_CREG_C1;
    sfpi::vConstFloatPrgm2 = SOFTPLUS_CREG_C2;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_softplus_body(const float beta, const float beta_reciprocal, const float threshold) {
    sfpi::vFloat val = sfpi::dst_reg[0];
    sfpi::vFloat t = beta * val;

    v_if(t <= threshold) {
        // a = |t| via setsgn (clear sign bit, no branch)
        sfpi::vFloat a = sfpi::setsgn(t, 0);

#ifdef INP_FLOAT32
        // FP32: f(a) via degree-8 Horner on [0, 5]
        sfpi::vFloat residual = PolynomialEvaluator::eval(
            a,
            sfpi::vConstFloatPrgm0,  // SOFTPLUS_POLY_C0, parked by softplus_init
            sfpi::vConstFloatPrgm1,  // SOFTPLUS_POLY_C1
            sfpi::vConstFloatPrgm2,  // SOFTPLUS_POLY_C2
            SOFTPLUS_POLY_C3,
            SOFTPLUS_POLY_C4,
            SOFTPLUS_POLY_C5,
            SOFTPLUS_POLY_C6,
            SOFTPLUS_POLY_C7,
            SOFTPLUS_POLY_C8);

        // Tail: f(a) ≈ exp(-a) for a > 5, via inline Cody-Waite exp +
        // 3-term Taylor ln(1+e) = e*(1 + e*(-1/2 + e/3))
        sfpi::vFloat neg_a = sfpi::setsgn(a, 1);
        v_if(a > SOFTPLUS_POLY_BOUNDARY) {
            sfpi::vFloat e = softplus_exp_negative(neg_a);
            residual = e * (1.0f + e * (-0.5f + e * 0.333333343f));
        }
        v_endif;
#else
        // BF16: f(a) via degree-6 Horner on [0, 5]
        sfpi::vFloat residual = PolynomialEvaluator::eval(
            a,
            sfpi::vConstFloatPrgm0,  // SOFTPLUS_BF16_POLY_C0, parked by softplus_init
            sfpi::vConstFloatPrgm1,  // SOFTPLUS_BF16_POLY_C1
            sfpi::vConstFloatPrgm2,  // SOFTPLUS_BF16_POLY_C2
            SOFTPLUS_BF16_POLY_C3,
            SOFTPLUS_BF16_POLY_C4,
            SOFTPLUS_BF16_POLY_C5,
            SOFTPLUS_BF16_POLY_C6);

        // Tail: the degree-6 poly diverges past its [0, 5] fit domain, while the true
        // residual < exp(-5) = 0.0067 there. Clamping to 0 keeps softplus(t>0) = t within
        // bf16 rounding and avoids the ~8-op exp tail on every element.
        v_if(a > SOFTPLUS_POLY_BOUNDARY) { residual = 0.0f; }
        v_endif;
#endif

        // Reconstruct softplus(t):
        //   t >= 0: softplus(t) = t + f(t) = max(0,t) + residual
        //   t < 0:  softplus(t) = f(|t|) = 0 + residual
        t = sfpi::max(t, 0.0f);
        sfpi::vFloat sp = t + residual;

        // Round-to-nearest for bf16 destination (SFPSTORE defaults to truncation)
        sfpi::vFloat result = beta_reciprocal * sp;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
    }
    v_endif;
}

#ifndef INP_FLOAT32
// Horner step of the bf16 residual polynomial, applied to two independent accumulators in
// lockstep. The steps must alternate at STATEMENT level: GCC will not interleave two
// independent expression trees on its own, so writing the two chains back to back leaves
// every dependency stall in place.
#define SOFTPLUS_STEP2(c)   \
    do {                    \
        r0 = r0 * a0 + (c); \
        r1 = r1 * a1 + (c); \
    } while (0)

// Two elements per iteration, hand-interleaved (bf16 path only -- the fp32 path's exp tail
// is a second, deeper chain that does not fit alongside a second element's live values, and
// the tt-llk perf sweep drives Float16_b, so an fp32 interleave could not be measured here).
//
// The residual polynomial is a fully dependent Horner chain: on Blackhole each step stalls
// on the previous SFPMAD with nothing to fill the slot, and the instruction count does not
// show it. Two independent chains fill those slots, and one dst_reg advance covers both.
//
// §2 cannot cross a v_if -- the condition-code state is one shared resource, so two
// elements taking different branches serialise their predicated blocks. The single-element
// body puts the whole polynomial inside `v_if (t <= threshold)`, so interleaving it in
// place would buy almost nothing. The fix is to lift the chain out of the guard rather
// than interleave within it (see the store comment below): dependent-adjacent pairs then
// fall 12 -> 6.5 per element, and loop cc does not grow (8 -> 7 per element, the nested
// PUSHC/POPC becoming two flat guards).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_softplus_body2(const float beta, const float beta_reciprocal, const float threshold) {
    sfpi::vFloat t0 = beta * sfpi::dst_reg[0];
    sfpi::vFloat t1 = beta * sfpi::dst_reg[1];

    // a = |t| via setsgn (clear sign bit, no branch)
    sfpi::vFloat a0 = sfpi::setsgn(t0, 0);
    sfpi::vFloat a1 = sfpi::setsgn(t1, 0);

    // f(a) via degree-6 Horner on [0, 5]. PolynomialEvaluator::eval expands to
    // `c + a * inner` from the innermost coefficient outwards, so descending C6..C0 with
    // `r = r * a + c` keeps each element's operations, operands and order: bit-exact.
    sfpi::vFloat r0 = SOFTPLUS_BF16_POLY_C6;
    sfpi::vFloat r1 = SOFTPLUS_BF16_POLY_C6;
    SOFTPLUS_STEP2(SOFTPLUS_BF16_POLY_C5);
    SOFTPLUS_STEP2(SOFTPLUS_BF16_POLY_C4);
    SOFTPLUS_STEP2(SOFTPLUS_BF16_POLY_C3);
    SOFTPLUS_STEP2(sfpi::vConstFloatPrgm2);  // SOFTPLUS_BF16_POLY_C2, parked by softplus_init
    SOFTPLUS_STEP2(sfpi::vConstFloatPrgm1);  // SOFTPLUS_BF16_POLY_C1
    SOFTPLUS_STEP2(sfpi::vConstFloatPrgm0);  // SOFTPLUS_BF16_POLY_C0

    // Tail: the degree-6 poly diverges past its [0, 5] fit domain; clamp to 0 as the
    // single-element body does.
    v_if(a0 > SOFTPLUS_POLY_BOUNDARY) { r0 = 0.0f; }
    v_endif;
    v_if(a1 > SOFTPLUS_POLY_BOUNDARY) { r1 = 0.0f; }
    v_endif;

    // Reconstruct softplus(t) = max(0, t) + f(|t|), then undo the beta scaling.
    sfpi::vFloat res0 = beta_reciprocal * (sfpi::max(t0, 0.0f) + r0);
    sfpi::vFloat res1 = beta_reciprocal * (sfpi::max(t1, 0.0f) + r1);

    // Round-to-nearest for bf16 destination (SFPSTORE defaults to truncation)
    if constexpr (!is_fp32_dest_acc_en) {
        res0 = sfpi::convert<sfpi::vFloat16b>(res0, sfpi::RoundMode::Nearest);
        res1 = sfpi::convert<sfpi::vFloat16b>(res1, sfpi::RoundMode::Nearest);
    }

    // The guard is applied only to the store, not to the arithmetic. SFPU predication masks
    // the destination *write* and not the execution, so evaluating the polynomial for every
    // lane and discarding it on the t > threshold lanes is bit-identical to the predicated
    // original -- those lanes keep the dst_reg value they came in with either way. This is
    // what makes the interleave above reachable: the independent work now sits in
    // straight-line code instead of inside a v_if it cannot cross.
    v_if(t0 <= threshold) { sfpi::dst_reg[0] = res0; }
    v_endif;
    v_if(t1 <= threshold) { sfpi::dst_reg[1] = res1; }
    v_endif;
}
#endif

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_softplus(std::uint32_t param0, std::uint32_t param1, std::uint32_t param2) {
    const float beta = Converter::as_float(param0);
    const float beta_reciprocal = Converter::as_float(param1);
    const float threshold = Converter::as_float(param2);
#ifdef INP_FLOAT32
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body<APPROXIMATION_MODE, is_fp32_dest_acc_en>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg++;
    }
#else
    for (int d = 0; d < ITERATIONS / 2; d++) {
        calculate_softplus_body2<APPROXIMATION_MODE, is_fp32_dest_acc_en>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg += 2;
    }
    // Odd ITERATIONS: finish the trailing element on the single-element body.
    if constexpr (ITERATIONS % 2 != 0) {
        calculate_softplus_body<APPROXIMATION_MODE, is_fp32_dest_acc_en>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg++;
    }
#endif
}

}  // namespace ckernel::sfpu

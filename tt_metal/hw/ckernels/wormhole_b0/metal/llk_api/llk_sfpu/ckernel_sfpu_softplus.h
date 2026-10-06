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
// BF16: branch-free evaluation in u = exp(-a):
//         f(a) = ln(1 + u) = u * (1 + u * h(u)),   u in (0, 1]
//       with h a degree-4 fit of (ln(1+u)/u - 1)/u on [0, 1]. Writing the
//       residual as u*(1 + u*h) makes it tend to u exactly as u -> 0, so the
//       negative tail stays relatively accurate down to the bf16 normal floor
//       instead of being clamped to 0. No tail branch, so no predicated exp
//       cost on every lane.
//       u = 2^(y-127), y = max(127 - a/ln2, 0). Wormhole has no truncating
//       fp32->uint16 conversion, so the caller passes yh = max(y - 1/2, 0) and
//       k = round-to-nearest(yh), which equals floor(y) for y >= 1/2. A
//       degree-3 polynomial for 2^(y-k) is placed under exponent k with one
//       SFPSETEXP. The max() is what sends y < 0 to k = 0: the conversion takes
//       the magnitude, so an unclamped negative y comes back as a positive k.
//       a > 88 clamps to yh = 0 and flushes to zero, the right bf16 answer.
//       Coefficients not held in the programmable constant registers are
//       fp16-representable so each costs one SFPLOADI, not two.
//       The main loop is an ILP unroll over two dest vectors per iteration
//       (each vector already spans two face rows) with the two chains
//       interleaved: the chain crosses SFPU units (MAD -> swap -> round ->
//       cast -> MAD ... -> setexp) and stalls when run alone. That stall
//       rationale and the figure below were measured on Blackhole; this
//       Wormhole variant (same coefficients, yh/round-to-nearest exponent
//       extraction) has been compiled, not run on silicon.
//       Measured on Blackhole over all 65536 bf16 encodings, scoring the
//       49,711 whose exact answer is a normal bf16: max 0.52 ULP end-to-end.
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

// BF16: -1/ln2 for y = 127 - a/ln2 (fp32, held in vConstFloatPrgm0), the fp32 exponent
// bias y is offset by, and the half the Wormhole path shifts y by so that
// round-to-nearest(yh) = floor(y): yh = y - HALF is formed and f = (yh - k) + HALF
// undoes it. The two uses must stay in step (see softplus_exp2_bf16).
constexpr float SOFTPLUS_BF16_NEG_ONE_LN2 = -1.4426950216293334961f;
constexpr float SOFTPLUS_BF16_EXP_BIAS = 127.0f;
constexpr float SOFTPLUS_BF16_HALF = 0.5f;

// BF16: p(f) ~ 2^f on [0, 1), p(f) = 1 + f*(P1 + f*(P2 + f*P3)), 1.05e-4 relative.
// p(0) = 1 exactly and p(1) = 2 - 2^-16 by construction (P1 = 1 - 2^-16 - P2 - P3), so
// p(f) stays inside [1, 2) and SFPSETEXP can place it under exponent k unchanged.
// P1 is fp32 (vConstFloatPrgm1); P0, P2, P3 are exactly representable in fp16.
constexpr float SOFTPLUS_BF16_P0 = 1.0f;
constexpr float SOFTPLUS_BF16_P1 = 0.6954193115234375f;
constexpr float SOFTPLUS_BF16_P2 = 0.2264404296875f;  // 1855 * 2^-13
constexpr float SOFTPLUS_BF16_P3 = 0.078125f;         // 5 * 2^-6

// BF16: h(u) = (ln(1+u)/u - 1)/u on [0, 1], degree 4 with h(0) = -1/2 pinned
// (1.3e-4 relative on h). H1 is fp32 (vConstFloatPrgm2); the rest are fp16-exact.
constexpr float SOFTPLUS_BF16_H0 = -0.5f;
constexpr float SOFTPLUS_BF16_H1 = 0.33147416f;
constexpr float SOFTPLUS_BF16_H2 = -0.229736328125f;    // -941 * 2^-12
constexpr float SOFTPLUS_BF16_H3 = 0.12548828125f;      // 257 * 2^-11
constexpr float SOFTPLUS_BF16_H4 = -0.03411865234375f;  // -559 * 2^-14

// ======================================================================
// Lightweight inline exp(x) for negative x (FP32 tail region).
// Adapted from gelu's x_times_exp_negative_tail (ckernel_sfpu_gelu.h).
// Uses Cody-Waite range reduction + Taylor polynomial (degree 7).
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

    // exp(r) via Taylor polynomial, |r| < 0.5, degree 7 for < 1 ULP
    sfpi::vFloat poly = PolynomialEvaluator::eval(
        r, 1.0f, 1.0f, 0.5f, 0.166666667f, 0.0416666667f, 0.00833333333f, 0.00138888889f, 0.000198412698f);

    // Scale by 2^k via exponent manipulation
    sfpi::vInt p_exp = sfpi::exexp(poly, sfpi::ExponentMode::Biased);
    sfpi::vInt new_exp = p_exp + k_int;

    // FTZ: if exponent underflows, result is 0
    sfpi::vFloat result = 0.0f;
    v_if(new_exp > 0) { result = sfpi::setexp(poly, new_exp); }
    v_endif;

    return result;
}

// ======================================================================
// BF16: u = 2^(y-127) for y = max(127 - a/ln2, 0), i.e. exp(-a) for a >= 0.
//   Wormhole has no truncating fp32->uint16 conversion, so the caller hands in
//   yh = max(y - 1/2, 0) and k = round-to-nearest(yh) = floor(y) for y >= 1/2.
//   f = y - k = (yh - k) + 1/2 in [0, 1), p(f) ~ 2^f in [1, 2), and
//   u = p(f) * 2^(k-127) is exactly setexp(p(f), k).
//   For a > 88 (exp(-a) below the bf16 normal range) yh = 0: k = 0 and p(1/2)
//   land under a zero exponent, i.e. a denormal the SFPU flushes to zero.
//   The caller's max(yh, 0) is required: the conversion takes the magnitude,
//   so an unclamped negative yh comes back as a positive k.
// ======================================================================
sfpi_inline sfpi::vFloat softplus_exp2_bf16(sfpi::vFloat yh) {
    sfpi::vUInt16 k = sfpi::convert<sfpi::vUInt16>(yh, sfpi::RoundMode::Nearest);
    sfpi::vFloat f = yh - sfpi::convert<sfpi::vFloat>(k, sfpi::RoundMode::Nearest);
    f = f + SOFTPLUS_BF16_HALF;
    sfpi::vFloat p =
        PolynomialEvaluator::eval(f, SOFTPLUS_BF16_P0, sfpi::vConstFloatPrgm1, SOFTPLUS_BF16_P2, SOFTPLUS_BF16_P3);
    return sfpi::setexp(p, sfpi::as<sfpi::vInt>(k));
}

// BF16 softplus for one vector, all lanes, no predication:
//   returns beta_reciprocal * softplus(t), rounded to bf16 unless dest is fp32.
// f(a) = ln(1+u) = u * (1 + u*h(u)), u = exp(-a), a = |t|. No tail branch.
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat softplus_bf16_eval(sfpi::vFloat t, const float beta_reciprocal) {
    sfpi::vFloat a = sfpi::setsgn(t, 0);
    // yh = y - 1/2 with y = 127 - a/ln2 (see softplus_exp2_bf16 for the rounding trick).
    sfpi::vFloat yh = a * sfpi::vConstFloatPrgm0 + (SOFTPLUS_BF16_EXP_BIAS - SOFTPLUS_BF16_HALF);
    // max(t, 0) is independent of yh; issued here it fills the bubble between
    // the MAD and the swap that clamps yh, and is long done when needed below.
    sfpi::vFloat sp = sfpi::max(t, 0.0f);
    yh = sfpi::max(yh, 0.0f);
    sfpi::vFloat u = softplus_exp2_bf16(yh);
    sfpi::vFloat h = PolynomialEvaluator::eval(
        u, SOFTPLUS_BF16_H0, sfpi::vConstFloatPrgm2, SOFTPLUS_BF16_H2, SOFTPLUS_BF16_H3, SOFTPLUS_BF16_H4);
    sfpi::vFloat q = u * h + 1.0f;

    // softplus(t) = max(t, 0) + f(|t|)
    sp = u * q + sp;

    sfpi::vFloat result = beta_reciprocal * sp;
    // Round-to-nearest for bf16 destination (SFPSTORE defaults to truncation)
    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    }
    return result;
}

// dst_reg vectors calculate_softplus consumes per iteration: two for the bf16 path
// (two vectors are evaluated per iteration so their dependent chains overlap), one for fp32.
#ifdef INP_FLOAT32
constexpr int SOFTPLUS_VECTORS_PER_ITER = 1;
#else
constexpr int SOFTPLUS_VECTORS_PER_ITER = 2;
#endif

// Resets the dest counters and loads the programmable constants. Each one holds an fp32
// constant that would otherwise cost two SFPLOADIs per use. The fp32 path (INP_FLOAT32)
// holds three of its degree-8 coefficients (all fp32, so any three save the same); the
// bf16 path holds the only three of its constants that are not fp16-exact.
inline void softplus_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);

#ifdef INP_FLOAT32
    sfpi::vConstFloatPrgm0 = SOFTPLUS_POLY_C0;
    sfpi::vConstFloatPrgm1 = SOFTPLUS_POLY_C2;
    sfpi::vConstFloatPrgm2 = SOFTPLUS_POLY_C4;
#else
    sfpi::vConstFloatPrgm0 = SOFTPLUS_BF16_NEG_ONE_LN2;
    sfpi::vConstFloatPrgm1 = SOFTPLUS_BF16_P1;
    sfpi::vConstFloatPrgm2 = SOFTPLUS_BF16_H1;
#endif
}

// Computes softplus for the vector at dst_reg[0] and stores it back in place.
// Lanes with t > threshold are left untouched by a predicated store, so they
// keep the input value (the identity result) without a separate select.
// Does not advance dst_reg; the caller does (calculate_softplus, SDPA).
// Both paths read vConstFloatPrgm0/1/2 loaded by softplus_init(). Call that
// init first, and again after another SFPU init overwrites those registers.
// SDPA's calculate_softplus_first_column uses this body and has that dependency.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_softplus_body(const float beta, const float beta_reciprocal, const float threshold) {
    sfpi::vFloat val = sfpi::dst_reg[0];
    sfpi::vFloat t = beta * val;

    // Everything sits under the threshold predicate: t is dead after the max()
    // swap (no copy needed) and the predicated store keeps the identity lanes.
    v_if(t <= threshold) {
#ifdef INP_FLOAT32
        // a = |t| via setsgn (clear sign bit, no branch)
        sfpi::vFloat a = sfpi::setsgn(t, 0);

        // FP32: f(a) via degree-8 Horner on [0, 5]
        sfpi::vFloat residual = PolynomialEvaluator::eval(
            a,
            sfpi::vConstFloatPrgm0,  // C0
            SOFTPLUS_POLY_C1,
            sfpi::vConstFloatPrgm1,  // C2
            SOFTPLUS_POLY_C3,
            sfpi::vConstFloatPrgm2,  // C4
            SOFTPLUS_POLY_C5,
            SOFTPLUS_POLY_C6,
            SOFTPLUS_POLY_C7,
            SOFTPLUS_POLY_C8);

        // Tail: f(a) ≈ exp(-a) for a > 5, via inline Cody-Waite exp +
        // 3-term Taylor ln(1+e) = e*(1 + e*(-1/2 + e/3))
        v_if(a > SOFTPLUS_POLY_BOUNDARY) {
            sfpi::vFloat e = softplus_exp_negative(-a);
            residual = e * (1.0f + e * (-0.5f + e * 0.333333343f));
        }
        v_endif;

        // Reconstruct softplus(t):
        //   t >= 0: softplus(t) = t + f(t) = max(0,t) + residual
        //   t < 0:  softplus(t) = f(|t|) = 0 + residual
        sfpi::vFloat sp = sfpi::max(t, 0.0f);
        sp = sp + residual;

        sfpi::vFloat result = beta_reciprocal * sp;
        // Round-to-nearest for bf16 destination (SFPSTORE defaults to truncation)
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
#else
        sfpi::vFloat result = softplus_bf16_eval<is_fp32_dest_acc_en>(t, beta_reciprocal);
#endif
        sfpi::dst_reg[0] = result;
    }
    v_endif;
}

// BF16: two vectors per iteration (dst_reg[0] and dst_reg[1]), hand-interleaved
// step by step so the two dependent chains overlap. Same u and q arithmetic as
// softplus_bf16_eval; PolynomialEvaluator is not used so the two Horner chains
// can be alternated. Keep the two copies in step: a coefficient or formula
// change has to land in both, since a shared helper would serialize the
// chains this interleave exists to overlap. Results are evaluated
// unpredicated; the predicated stores then keep the identity lanes. Does not
// advance dst_reg; the caller does. Reads vConstFloatPrgm0/1/2 from softplus_init().
// The lreg budget (L0-L7) is tight, so only |t| is kept up front and x is
// re-read from dest where needed (a load is cheaper than a live lreg here).
// beta arrives as a vFloat the caller loads once, outside the loop: it is read
// four times per iteration, and as a runtime float each read is two SFPLOADIs.
// It takes the last free lreg. positive_beta picks the linear term (see below).
template <bool is_fp32_dest_acc_en, bool positive_beta>
sfpi_inline void softplus_body_bf16_x2(const sfpi::vFloat beta, const float beta_reciprocal, const float threshold) {
    // Constants are materialised once per iteration and shared by both vectors.
    // yh = y - 1/2 (see softplus_exp2_bf16 for the Wormhole rounding trick).
    sfpi::vFloat cBiasMinusHalf = SOFTPLUS_BF16_EXP_BIAS - SOFTPLUS_BF16_HALF;
    sfpi::vFloat y0 = sfpi::setsgn(beta * sfpi::dst_reg[0], 0) * sfpi::vConstFloatPrgm0 + cBiasMinusHalf;
    sfpi::vFloat y1 = sfpi::setsgn(beta * sfpi::dst_reg[1], 0) * sfpi::vConstFloatPrgm0 + cBiasMinusHalf;
    y0 = sfpi::max(y0, 0.0f);
    y1 = sfpi::max(y1, 0.0f);
    sfpi::vUInt16 k0 = sfpi::convert<sfpi::vUInt16>(y0, sfpi::RoundMode::Nearest);
    sfpi::vUInt16 k1 = sfpi::convert<sfpi::vUInt16>(y1, sfpi::RoundMode::Nearest);
    sfpi::vFloat cHalf = SOFTPLUS_BF16_HALF;
    sfpi::vFloat f0 = y0 - sfpi::convert<sfpi::vFloat>(k0, sfpi::RoundMode::Nearest);
    sfpi::vFloat f1 = y1 - sfpi::convert<sfpi::vFloat>(k1, sfpi::RoundMode::Nearest);
    f0 = f0 + cHalf;
    f1 = f1 + cHalf;

    sfpi::vFloat cP3 = SOFTPLUS_BF16_P3;
    sfpi::vFloat cP2 = SOFTPLUS_BF16_P2;
    sfpi::vFloat p0 = f0 * cP3 + cP2;
    sfpi::vFloat p1 = f1 * cP3 + cP2;
    p0 = f0 * p0 + sfpi::vConstFloatPrgm1;
    p1 = f1 * p1 + sfpi::vConstFloatPrgm1;
    p0 = f0 * p0 + SOFTPLUS_BF16_P0;
    p1 = f1 * p1 + SOFTPLUS_BF16_P0;
    sfpi::vFloat u0 = sfpi::setexp(p0, sfpi::as<sfpi::vInt>(k0));
    sfpi::vFloat u1 = sfpi::setexp(p1, sfpi::as<sfpi::vInt>(k1));

    sfpi::vFloat cH4 = SOFTPLUS_BF16_H4;
    sfpi::vFloat cH3 = SOFTPLUS_BF16_H3;
    sfpi::vFloat h0 = u0 * cH4 + cH3;
    sfpi::vFloat h1 = u1 * cH4 + cH3;
    sfpi::vFloat cH2 = SOFTPLUS_BF16_H2;
    h0 = u0 * h0 + cH2;
    h1 = u1 * h1 + cH2;
    h0 = u0 * h0 + sfpi::vConstFloatPrgm2;
    h1 = u1 * h1 + sfpi::vConstFloatPrgm2;
    sfpi::vFloat cH0 = SOFTPLUS_BF16_H0;
    h0 = u0 * h0 + cH0;
    h1 = u1 * h1 + cH0;
    h0 = u0 * h0 + 1.0f;  // q = 1 + u*h
    h1 = u1 * h1 + 1.0f;

    // softplus(x) = (max(t, 0) + u*q) / beta = max(t, 0)/beta + u*(q/beta). The linear
    // term is max(x, 0) for beta > 0 and min(x, 0) for beta < 0 (t = beta*x flips sign),
    // so it needs neither beta nor beta_reciprocal: two multiplies fewer per vector, and
    // the beta * beta_reciprocal round trip on the linear term is gone (bit-identical to
    // the single-vector form when beta == 1). x is re-read from dest (the swap consumes
    // its input, and the predicates below re-read it once more). beta_reciprocal is
    // materialised once and shared by both vectors.
    sfpi::vFloat cRecip = beta_reciprocal;
    h0 = h0 * cRecip;
    h1 = h1 * cRecip;
    sfpi::vFloat m0;
    sfpi::vFloat m1;
    if constexpr (positive_beta) {
        m0 = sfpi::max(sfpi::dst_reg[0], 0.0f);
        m1 = sfpi::max(sfpi::dst_reg[1], 0.0f);
    } else {
        m0 = sfpi::min(sfpi::dst_reg[0], 0.0f);
        m1 = sfpi::min(sfpi::dst_reg[1], 0.0f);
    }
    sfpi::vFloat r0 = u0 * h0 + m0;
    sfpi::vFloat r1 = u1 * h1 + m1;
    // Round-to-nearest for bf16 destination (SFPSTORE defaults to truncation)
    if constexpr (!is_fp32_dest_acc_en) {
        r0 = sfpi::convert<sfpi::vFloat16b>(r0, sfpi::RoundMode::Nearest);
        r1 = sfpi::convert<sfpi::vFloat16b>(r1, sfpi::RoundMode::Nearest);
    }
    sfpi::vFloat thr = threshold;
    v_if(beta * sfpi::dst_reg[0] <= thr) { sfpi::dst_reg[0] = r0; }
    v_endif;
    v_if(beta * sfpi::dst_reg[1] <= thr) { sfpi::dst_reg[1] = r1; }
    v_endif;
}

// BF16 main loop. beta is loaded into an lreg once here and stays live across the
// iterations (softplus_body_bf16_x2 explains why); the sign of beta is decided once,
// on the scalar side, so the loop body carries no per-iteration select.
template <bool is_fp32_dest_acc_en, bool positive_beta, int ITERATIONS>
sfpi_inline void softplus_loop_bf16_x2(const float beta, const float beta_reciprocal, const float threshold) {
    static_assert(ITERATIONS % SOFTPLUS_VECTORS_PER_ITER == 0);
    const sfpi::vFloat beta_v = beta;
    for (int d = 0; d < ITERATIONS / SOFTPLUS_VECTORS_PER_ITER; d++) {
        softplus_body_bf16_x2<is_fp32_dest_acc_en, positive_beta>(beta_v, beta_reciprocal, threshold);
        sfpi::dst_reg += SOFTPLUS_VECTORS_PER_ITER;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_softplus(std::uint32_t param0, std::uint32_t param1, std::uint32_t param2) {
    const float beta = Converter::as_float(param0);
    const float beta_reciprocal = Converter::as_float(param1);
    const float threshold = Converter::as_float(param2);
#ifdef INP_FLOAT32
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body<APPROXIMATION_MODE, is_fp32_dest_acc_en>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg += SOFTPLUS_VECTORS_PER_ITER;
    }
#else
    if (beta > 0.0f) {
        softplus_loop_bf16_x2<is_fp32_dest_acc_en, true, ITERATIONS>(beta, beta_reciprocal, threshold);
    } else {
        softplus_loop_bf16_x2<is_fp32_dest_acc_en, false, ITERATIONS>(beta, beta_reciprocal, threshold);
    }
#endif
}

}  // namespace ckernel::sfpu

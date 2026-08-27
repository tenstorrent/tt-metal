// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// ======================================================================
// i0(x) — modified Bessel function of the first kind, order 0.
//
// Two-region implementation, exploiting that i0 is even: i0(-x) = i0(x).
//   |x| <= 6:  Maclaurin series, I0(x) = sum_k (x^2/4)^k / (k!)^2,
//              degree-8 minimax fit in t=x^2 (9 terms incl. the leading
//              1.0), Remez-derived. Max rel err ~3.1e-07 in idealized
//              arithmetic, dominated in practice by rounding t = fl(x*x)
//              itself rather than by the polynomial fit — see the note
//              below.
//   |x| >  6:  asymptotic expansion (Abramowitz & Stegun 9.7.1)
//                i0(x) = exp(|x|) / sqrt(|x|) * Q(1/|x|)
//              FP32: degree-5 fit (6 coeffs). BF16: degree-4 fit (5
//              coeffs) — one term shorter, since BF16 output precision
//              cannot see the extra term (0/498 exhaustive BF16 values
//              in [6, 88.5] differ from the correctly-rounded result).
//              1/sqrt(2*pi) folded into Q's leading term in both.
//
// Measured worst case over the full domain (200k-sample random sweep over
// [-88.5, 88.5], both dtypes, same seed on both arches): 6.0 FP32 ULP (at
// x = -5.93, inside region 1) and 0.91 BF16 ULP. Blackhole p150b and
// Wormhole n300 silicon produced bit-identical results -- same worst-case
// x, same output, same reference, to the last digit -- so the WH/BH source
// identity this file maintains is measured equivalence, not an inference
// from it. Identical to the pre-trim kernel's FP32 figure and marginally
// better on BF16 -- trimming both regions did not measurably change
// accuracy, confirming the analysis below. See tests/.../test_unary_i0.py's
// _MAX_ULP for the test budget (12 / 2), which already carried headroom
// over these numbers.
//
// Both regions were originally 3 terms longer (12-term Maclaurin, one
// shared 6-term Q for both dtypes). The trim follows a silicon+analysis
// review (tenstorrent/tt-metal#52126, comment 5265986150): compiling
// standalone against the pinned sfpi build and counting the disassembly,
// plus bit-exact FP32/BF16 Horner simulation against mpmath, showed the
// extra terms cost real instructions (dominated by SFPLOADI coefficient
// materialization) without measurable accuracy — because accuracy near
// the region-1 boundary is bounded by FP32 rounding of t = fl(x*x), not
// by truncation, and the region-2 rsqrt Newton step alone already costs
// ~1.7 FP32 ULP, both floors well above what the trimmed terms bought.
// This kernel's own independent Remez fit and FP32/BF16 simulation (see
// tt-bounty/tt-metal/04/review-52126 for the scripts) reproduced those findings.
//
// Code shape (chosen to relieve SFPI LRA budget), mirroring i1:
//   1. Compute polynomial result unconditionally and store to DST.
//      Polynomial-path intermediates die at the store, freeing LRegs.
//   2. v_if (|x|>6): overwrite DST with asymptotic result.
//
// No input clamp: unlike the previous version, abs_x here is never
// clamped to 88.5 before use. Every lane a clamp would change is a lane
// with |x| > 88.5, and every one of those is unconditionally overwritten
// below by the overflow branch — so a clamp only ever protects a value
// that is then discarded. The unclamped polynomial/asymptotic paths can
// produce garbage (even overflow to inf, or NaN for NaN input) on those
// lanes; since SFPU lanes are independent and the garbage never reaches
// the store, this is safe and costs nothing.
//
// Overflow, +/-inf and NaN all resolve in one branch: multiplying by
// infinity rather than assigning it handles all three cases from a
// single predicate. Finite |x| > 88.5 gives +inf (FP32 exp() saturates
// at 88.7228, matching torch's identical limitation — cf. the note on
// torch.sinh in test_unary_fp32.py); +/-inf gives +inf; NaN stays NaN,
// relying on SFPMUL propagating a NaN operand the same way
// _sfpu_exp_fp32_accurate_ relies on 0*inf = NaN for the same purpose.
// The true I0 stays representable to x = 91.9008 (I0 = 3.4028e+38), but
// no exp()-first formulation can reach it without the intermediate
// overflowing.
//
// On the BF16 path NaN still emerges as +inf regardless of this branch:
// a DRAM round-trip with no op returns NaN intact, so the payload is
// lost unpacking BF16 into DST, upstream of this kernel entirely.
//
// APPROXIMATION_MODE is accepted for call-site compatibility; both paths
// are already the accurate ones and it does not select a cheaper route.
// ======================================================================

// Asymptotic path is outlined to keep register pressure within SFPI's
// LRA budget. Returns exp(|x|) * 1/sqrt(|x|) * Q(1/|x|).
// Note: this function must stay minimalist — SFPU LRA is limited.
// Every operation here competes with the main loop.
inline sfpi::vFloat calculate_i0_asymptotic_(const sfpi::vFloat abs_x) {
    // exp(|x|) — unsafe variants in both paths. For the |x| in [6, 88.5] that
    // reaches the store, overflow/underflow is impossible and the safe
    // wrappers' clamping/guards would be dead code. Above 88.5 these wrap and
    // return garbage; the caller overwrites those lanes unconditionally.
#ifdef INP_FLOAT32
    const sfpi::vFloat exp_abs = _sfpu_exp_fp32_accurate_unsafe_(abs_x);
#else
    const sfpi::vFloat exp_abs = _sfpu_exp_21f_bf16_unsafe_<true>(abs_x);
#endif

    // 1/sqrt(|x|) via Quake-style magic constant + two Newton refinements.
    // Computed first so that 1/|x| can be derived as rsqrt_y^2 without a
    // separate sfpu_reciprocal call.
    //
    // Kept unconditional (not dtype-split): the Newton step is load-bearing
    // for FP32 (dropping it costs ~343 FP32 ULP against a budget in the
    // teens) and only marginal for BF16 (~0.2% of outputs move by 1 ULP,
    // comfortably inside the existing 2-ULP BF16 test budget either way).
    // Splitting it to save 4 instructions on the BF16 path was left alone: it
    // would add a third dtype branch inside the kernel's most precision-
    // sensitive block for the smallest of the four savings on the table.
    const sfpi::vInt rsqrt_i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(abs_x) >> 1);
    sfpi::vFloat rsqrt_y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - rsqrt_i);
    sfpi::vFloat c0 = (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = rsqrt_y * (sfpi::vFloat(2.2825186f) + c0 * (sfpi::vFloat(2.2533049f) + c0));
    c0 = 1.0f + (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = c0 * sfpi::addexp(rsqrt_y, -1) + rsqrt_y;

    // 1/|x| = (1/sqrt(|x|))^2 — reuses the refined rsqrt instead of a fresh reciprocal.
    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;

    // Q(y) on y in [1/88.5, 1/6]; leading term is 1/sqrt(2*pi) = 0.39894228.
    // This outlined function does not stress the main loop's LRA, so full
    // precision is safe.
    //
    // FP32: degree-5 (6 coeffs). BF16: degree-4 (5 coeffs) — one term
    // shorter, verified against an exhaustive sweep of all BF16-representable
    // values in [6, 88.5] (0/498 differ from the correctly-rounded result).
    // Coefficients must stay FP32 in both cases: rounding the BF16 Q's
    // leading term to BF16 (which would let sfpi use SFPADDI's 16-bit
    // immediate instead of an SFPLOADI pair) costs 1.27e-03 relative on that
    // term alone — four orders of magnitude above the accuracy budget.
#ifdef INP_FLOAT32
    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x,
        3.9894214272e-01f,
        4.9887448549e-02f,
        2.7172168717e-02f,
        4.6332854778e-02f,
        -1.0997270793e-01f,
        6.5736579895e-01f);
#else
    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x, 3.9894300699e-01f, 4.9790032208e-02f, 3.0512193218e-02f, -9.7926484887e-04f, 1.8299004436e-01f);
#endif

    // i0 is even — no sign restoration needed (cf. i1's copysgn).
    return exp_abs * rsqrt_y * correction;
}

inline void i0_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    constexpr float I0_MAX_INPUT = 88.5f;
    constexpr float I0_THRESHOLD = 6.0f;

    // Decorative but intentional: unroll 0, unroll 1, and no pragma at all
    // produce byte-identical codegen here (-funroll-loops included), since
    // sfpi lowers this loop's body through the 32-entry Tensix replay buffer
    // regardless — dynamic work per datum stays flat all the way to unroll
    // 8, while code size grows ~5.7x. Kept as documentation of that, not as
    // a knob that does anything.
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        // i0 is even, so the sign is never needed: take |x| up front with a
        // plain abs. No clamp — see the file-level comment for why an
        // unclamped magnitude here is safe.
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat abs_x = sfpi::abs(x);

        sfpi::vFloat val;
        // ─── Polynomial path (always; valid for |x| <= 6) ────────────────
        // Computed unconditionally so its LRegs are free for the asymptotic
        // block to reuse.
        //
        // Degree-8 minimax fit in t (Remez, weighted for relative I0 error),
        // not truncated Maclaurin — a naive truncation at this degree is
        // measurably worse. k = 1 and k = 2 are pinned to their exact
        // Maclaurin values (1/4 and 1/64): both are BF16-representable, so
        // sfpi folds them into SFPADDI's immediate instead of materialising
        // them with an SFPLOADI pair.
        {
            const sfpi::vFloat t = abs_x * abs_x;
            val = 1.0f + t * PolynomialEvaluator::eval(
                                 t,
                                 2.5000000000e-01f,
                                 1.5625000000e-02f,
                                 4.3402606389e-04f,
                                 6.7823571044e-06f,
                                 6.7718225694e-08f,
                                 4.7797393821e-10f,
                                 2.1436320601e-12f,
                                 1.4051053479e-14f);
        }

        // ─── Asymptotic overwrite for OOD lanes (|x| > 6) ────────────────
        v_if(abs_x > I0_THRESHOLD) { val = calculate_i0_asymptotic_(abs_x); }
        v_endif;

        // ─── Overflow, +/-inf and NaN → +inf (NaN stays NaN) ─────────────
        // See the file-level comment for the SFPMUL/NaN-propagation
        // assumption this relies on, and why it replaces two branches
        // (overflow-assign + bit-pattern NaN-restore) with one.
        v_if(abs_x > I0_MAX_INPUT) { val = abs_x * std::numeric_limits<float>::infinity(); }
        v_endif;
#ifndef INP_FLOAT32
        val = sfpi::convert<sfpi::vFloat16b>(val, sfpi::RoundMode::Nearest);
#endif
        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

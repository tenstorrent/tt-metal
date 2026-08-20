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
//              truncated at k=11 (12 terms). Max rel err 1.4e-08.
//   |x| >  6:  asymptotic expansion (Abramowitz & Stegun 9.7.1)
//                i0(x) = exp(|x|) / sqrt(|x|) * Q(1/|x|)
//              degree-5 fit (6 coeffs), 1/sqrt(2*pi) folded into Q's
//              leading term. Max rel err 4.7e-08 over [6, 88.5].
//
// Worst case over [0, 88.5] is 4.7e-08 relative (~0.4 FP32 ULP), at the
// handover. The threshold was chosen by sweeping it: the polynomial's
// truncation grows and the asymptotic fit's residual shrinks with |x|,
// and 6 is where the two curves cross. Above ~13 the polynomial alone
// (the previous implementation) is unusable: 21% rel err at x=20, 89%
// at x=30, because the series' largest term sits near k ~ x/2 and the
// omitted tail — eventually the omitted peak itself — dominates.
//
// Code shape (chosen to relieve SFPI LRA budget), mirroring i1:
//   1. Compute polynomial result unconditionally and store to DST.
//      Polynomial-path intermediates die at the store, freeing LRegs.
//   2. v_if (|x|>6): overwrite DST with asymptotic result.
//
// Overflow: I0 grows without bound, so |x| past the representable range
// returns +inf rather than a clamped finite value. The computation is
// still clamped to 88.5 (FP32 exp() saturates at 88.7228), and lanes above
// that bound are then overwritten with +inf. i0(+/-inf) = +inf follows from
// the same test.
//
// This matches ttnn's declared golden, torch.i0, which overflows to inf at
// the same 88.7228 for the same reason (cf. the note on torch.sinh in
// test_unary_fp32.py). The two differ only for FP32 inputs in the 0.22-wide
// window (88.5, 88.7228), where torch still returns a finite value below
// 1.45e+37; no bfloat16 value lands strictly inside it. The true I0 stays
// representable to x = 91.9008 (I0 = 3.4028e+38), but no exp()-first
// formulation can reach it without the intermediate overflowing.
//
// APPROXIMATION_MODE is accepted for call-site compatibility; both paths
// are already the accurate ones and it does not select a cheaper route.
// ======================================================================

// Asymptotic path is outlined to keep register pressure within SFPI's
// LRA budget. Returns exp(|x|) * 1/sqrt(|x|) * Q(1/|x|).
// Note: this function must stay minimalist — SFPU LRA is limited.
// Every operation here competes with the main loop.
inline sfpi::vFloat calculate_i0_asymptotic_(const sfpi::vFloat abs_x) {
    // exp(|x|) — unsafe variants in both paths: |x| in [6, 88.5] precludes
    // overflow/underflow, so the safe wrappers' clamping/guards are dead
    // and skipped.
#ifdef INP_FLOAT32
    const sfpi::vFloat exp_abs = _sfpu_exp_fp32_accurate_unsafe_(abs_x);
#else
    const sfpi::vFloat exp_abs = _sfpu_exp_21f_bf16_unsafe_<true>(abs_x);
#endif

    // 1/sqrt(|x|) via Quake-style magic constant + two Newton refinements.
    // Computed first so that 1/|x| can be derived as rsqrt_y^2 without a
    // separate sfpu_reciprocal call.
    const sfpi::vInt rsqrt_i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(abs_x) >> 1);
    sfpi::vFloat rsqrt_y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - rsqrt_i);
    sfpi::vFloat c0 = (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = rsqrt_y * (sfpi::vFloat(2.2825186f) + c0 * (sfpi::vFloat(2.2533049f) + c0));
    c0 = 1.0f + (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = c0 * sfpi::addexp(rsqrt_y, -1) + rsqrt_y;

    // 1/|x| = (1/sqrt(|x|))^2 — reuses the refined rsqrt instead of a fresh reciprocal.
    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;

    // Q(y) on y in [1/88.5, 1/6]; leading term is 1/sqrt(2*pi) = 0.39894228.
    // This outlined function does not stress the main loop's LRA, so full precision is safe.
    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x,
        3.9894214272e-01f,
        4.9887448549e-02f,
        2.7172168717e-02f,
        4.6332854778e-02f,
        -1.0997270793e-01f,
        6.5736579895e-01f);

    // i0 is even — no sign restoration needed (cf. i1's copysgn).
    return exp_abs * rsqrt_y * correction;
}

inline void i0_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    constexpr float I0_MAX_INPUT = 88.5f;
    constexpr float I0_THRESHOLD = 6.0f;

#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        // i0 is even, so the sign is never needed: take |x| up front and clamp with a
        // plain min. Strictly cheaper than symmetric_clamp() followed by abs(), which
        // computes a copysgn() only for the abs() to discard it — and it leaves the
        // unclamped magnitude live for the overflow test below at no extra LReg cost.
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat abs_x_in = sfpi::abs(x);

        // Clamp to 88.5 — exp() saturates near 88.7 in FP32.
        const sfpi::vFloat abs_x = sfpi::min(abs_x_in, I0_MAX_INPUT);

        sfpi::vFloat val;
        // ─── Polynomial path (always; valid for |x| <= 6) ────────────────
        // Computed unconditionally and stored — its LRegs are then free
        // for the asymptotic block to use.
        //
        // Coefficients are (1/4)^k/(k!)^2 for k = 1..11, rounded to FP32.
        // The previous implementation carried these to only three
        // significant figures, which cost ~52 FP32 ULP at |x| = 5 on its
        // own — independent of, and masked by, the truncation problem.
        {
            const sfpi::vFloat t = abs_x * abs_x;
            val = 1.0f + t * PolynomialEvaluator::eval(
                                 t,
                                 2.5000000000e-01f,
                                 1.5625000000e-02f,
                                 4.3402778101e-04f,
                                 6.7816840783e-06f,
                                 6.7816841920e-08f,
                                 4.7095027877e-10f,
                                 2.4028075536e-12f,
                                 9.3859670063e-15f,
                                 2.8969033031e-17f,
                                 7.2422583611e-20f,
                                 1.4963343367e-22f);
        }

        // ─── Asymptotic overwrite for OOD lanes (|x| > 6) ────────────────
        v_if(abs_x > I0_THRESHOLD) { val = calculate_i0_asymptotic_(abs_x); }
        v_endif;

        // ─── Overflow: |x| past the representable range → +inf ───────────
        // Tested on the *unclamped* magnitude. Without this, i0(1e4) would
        // return i0(88.5) = 1.16e+37 — a silent wrong finite answer.
        // i0(+/-inf) = +inf falls out of the same test.
        v_if(abs_x_in > I0_MAX_INPUT) { val = std::numeric_limits<float>::infinity(); }
        v_endif;

        // ─── NaN in → NaN out ────────────────────────────────────────────
        // The SFPU compare is not IEEE-ordered: NaN carries the maximal
        // exponent and passes the > test above, so it would be converted to
        // +inf. Detect it by bit pattern instead — exponent all ones with a
        // non-zero mantissa — and restore it.
        //
        // Measured on Blackhole p150b: this restores NaN on the FP32 path.
        // On the BF16 path NaN still emerges as +inf — a DRAM round-trip with
        // no op returns NaN intact, so the payload is lost unpacking BF16 into
        // DST, upstream of this branch. Both remain an improvement on the
        // previous behaviour, where the input clamp mapped NaN to a finite
        // 1.15e+37.
        v_if(sfpi::exexp(abs_x_in) == 128 && sfpi::exman(abs_x_in) != 0) { val = abs_x_in; }
        v_endif;
#ifndef INP_FLOAT32
        val = sfpi::convert<sfpi::vFloat16b>(val, sfpi::RoundMode::Nearest);
#endif
        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

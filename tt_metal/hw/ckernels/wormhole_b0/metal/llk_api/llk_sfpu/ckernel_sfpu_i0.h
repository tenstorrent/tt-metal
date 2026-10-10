// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// ======================================================================
// i0(x) — modified Bessel function of the first kind, order 0.
//
// Two regions, exploiting that i0 is even: i0(-x) = i0(x).
//   |x| <= 6:  I0 = 1 + t * P(t), t = x^2, P a minimax fit weighted for
//              relative I0 error.
//   |x| >  6:  I0 = exp(|x|) / sqrt(|x|) * Q(1/|x|)  (Abramowitz & Stegun
//              9.7.1), with 1/sqrt(2*pi) folded into Q's leading term.
//
// The SFPU executes both regions for every vector, so a tile costs the sum
// of the two; each piece is sized to the precision the output can hold, and
// all of them key on is_fp32_dest_acc_en (the DEST width), never on
// INP_FLOAT32 -- a bf16-in/float32-out call runs a 32-bit DEST with
// INP_FLOAT32 undefined, and its output must keep float32 precision.
//
//                32-bit DEST                      16-bit DEST
//   P(t)         degree 8 in t                    degree 5 in t
//   exp(|x|)     Cody-Waite + degree-6 poly (*)   bit pattern of |x|/ln2 + 383,
//                                                 cubic correction
//   1/sqrt(|x|)  magic seed, cubic correction,    magic seed, cubic correction
//                one Newton step
//   Q(u)         degree 5                         degree 1
//   (*) _sfpu_exp_fp32_accurate_nonneg_unsafe_, the ckernel_sfpu_exp.h
//       variant for 0 <= x < 88.72.
//
// Cost is set by instruction count plus stalls, not by FLOPs: an SFPLOADI
// pair materialises each float32 constant, and Blackhole stalls an
// instruction that reads the result of the SFPMAD issued just before it
// (Wormhole pads that slot with an SFPNOP). Hence the shape below:
//   - every polynomial coefficient that accuracy allows is bf16- or
//     fp16-exact, so it loads with one SFPLOADI instead of two (constrained
//     minimax fits);
//   - the rsqrt seed constant and the two cubic-correction constants live
//     in the programmable constant registers, set once in i0_init();
//   - the 1/sqrt(|x|) chain is interleaved with region 1's Horner chain, so
//     each fills the other's SFPMAD latency slots. The interleave changes
//     no arithmetic (outputs measured bit-identical to the sequential order)
//     and takes 7% off the per-tile time on Blackhole p150a in both modes.
//
// Measured on Blackhole p150a: 0.6391 bf16 ULP worst case over every
// bfloat16 input against the float32 golden (16-bit DEST); 6 float32 ULP
// over a 2,000,001-point sweep of [-88.5, 88.5] (32-bit DEST). See
// tests/.../test_unary_i0.py's _MAX_ULP for the test budgets (12 / 2).
//
// Region 1 on a 32-bit DEST is limited by float32 rounding of t = fl(x*x)
// and of the Horner chain, not by the fit (1.0e-8 relative): the worst case
// sits just below the |x| = 6 split.
//
// No input clamp: abs_x is never clamped to 88.5 before use. Every lane a
// clamp would change has |x| > 88.5, and every one of those is
// unconditionally overwritten below by the overflow branch -- so a clamp
// only ever protects a value that is then discarded. Both exp
// constructions assume |x| <= 88.5 (they assemble the exponent field
// directly) and produce garbage above it; SFPU lanes are independent and
// that garbage never reaches the store.
//
// Overflow and +/-inf both resolve to +inf here: multiplying by infinity
// rather than assigning it lets one predicate cover both. The 88.5 cutoff
// is Q's fitted boundary, not exp()'s true saturation point (88.7228) -- a
// narrow finite, torch-matching band above 88.5 also returns +inf here,
// accepted rather than re-fit since it is invisible in BF16 and already deep
// in i0's exponential growth. NaN survives regardless of which branch it
// takes: lanes the compare excludes propagate NaN through ordinary
// arithmetic in region 1, lanes it includes propagate NaN through this
// SFPMUL (0*inf and NaN*inf are NaN) -- so correctness does not depend on
// how SFPSETCC orders NaN against the threshold. The true I0 stays
// representable to x = 91.9008 (I0 = 3.4028e+38), but no exp()-first
// formulation can reach it without the intermediate overflowing.
//
// On the BF16 path NaN still emerges as +inf regardless of this branch: a
// DRAM round-trip with no op returns NaN intact, so the payload is lost
// unpacking BF16 into DST, upstream of this kernel entirely.
//
// APPROXIMATION_MODE is accepted for call-site compatibility; both paths
// are already the accurate ones and it does not select a cheaper route.
// ======================================================================

// 16-bit DEST: exp(|x|) * rsqrt_y * Q(rsqrt_y^2), with rsqrt_y = 1/sqrt(|x|)
// from the caller, ordered so independent work fills SFPMAD latency slots.
// 1/|x| = (1/sqrt(|x|))^2 reuses the rsqrt instead of a fresh reciprocal;
// Q(u) is fit on u in [1/88.5, 1/6] for relative error. The 32-bit DEST
// counterpart is written out in calculate_i0.
inline sfpi::vFloat calculate_i0_asymptotic_bf16_(const sfpi::vFloat abs_x, const sfpi::vFloat rsqrt_y) {
    // exp(|x|) from the bit pattern of w = |x|/ln2 + 383: for |x| in
    // [0, 89.4], w lies in [383, 512), so its exponent is fixed and its 23
    // fraction bits hold (|x|/ln2 + 127) * 2^15. Shifted left by 8 they
    // read as a float32 z = 2^n * (1 + r), n = floor(|x|/ln2), r the
    // fraction to 15 bits -- the integer split costs two instructions.
    // exp(|x|) = z * C(m), C(m) = 2^(m-1)/m with m = 1 + r, a cubic fit
    // (8.7e-4 relative, all four coefficients fp16-exact).
    const sfpi::vFloat w = abs_x * 1.442695f + 383.0f;
    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;
    const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(sfpi::exman(w)) << 8);
    const sfpi::vFloat c = PolynomialEvaluator::eval(
        sfpi::setexp(z, 127), 1.775390625f, -1.376953125f, 0.70751953125f, -0.10650634765625f);
    // Degree 1, fit 4.9e-4 relative, both coefficients fp16-exact.
    const sfpi::vFloat correction = inv_abs_x * 0.05609130859375f + 0.398681640625f;
    return z * c * rsqrt_y * correction;
}

// The rsqrt seed constant and its cubic-correction constants, shared by both
// DEST widths. Programmable registers hold them across the loop, where each
// would otherwise be an SFPLOADI pair per iteration.
inline void i0_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstIntPrgm0 = 0x5f1110a0;
    sfpi::vConstFloatPrgm1 = 2.2825186f;
    sfpi::vConstFloatPrgm2 = 2.2533049f;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_i0() {
    constexpr float I0_MAX_INPUT = 88.5f;
    constexpr float I0_THRESHOLD = 6.0f;

    // Decorative but intentional: unroll 0, unroll 1, and no pragma at all
    // produce byte-identical codegen here (-funroll-loops included) --
    // dynamic work per datum stays flat all the way to unroll 8, while code
    // size grows ~5.7x. Kept as documentation of that, not as a knob that
    // does anything.
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        // i0 is even, so the sign is never needed: take |x| up front with a
        // plain abs. No clamp -- see the file-level comment for why an
        // unclamped magnitude here is safe.
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat abs_x = sfpi::abs(x);

        // ─── Region 1 (always; valid for |x| <= 6) and 1/sqrt(|x|) ──────────
        // Computed unconditionally, in a nested scope so their temporaries
        // are freed before the asymptotic block needs the LRegs.
        //
        // P is evaluated by Horner (p) with the 1/sqrt(|x|) chain (rsqrt_y)
        // interleaved statement by statement; see the file-level comment.
        // 1/sqrt(|x|): magic seed vConstIntPrgm0 - (bits >> 1), then the cubic
        // correction y * (k1 + c * (k2 + c)), c = -|x| * y^2 (2.0e-5 relative);
        // a 32-bit DEST adds one Newton step (7.3e-8), without which the
        // result is ~343 float32 ULP off.
        sfpi::vFloat val;
        sfpi::vFloat rsqrt_y;
        sfpi::vFloat q;  // Q(1/|x|), 32-bit DEST only
        {
            const sfpi::vFloat t = abs_x * abs_x;
            const sfpi::vInt rsqrt_i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(abs_x) >> 1);
            if constexpr (is_fp32_dest_acc_en) {
                // Degree 8 in t (9 terms incl. the leading 1), 1.0e-8 relative.
                // k = 1, 2 are the exact Maclaurin values 1/4 and 1/64; k = 7, 8
                // are bf16-exact, with the rest refit around them.
                sfpi::vFloat p = t * 1.4044321e-14f + 2.145839e-12f;
                rsqrt_y = sfpi::as<sfpi::vFloat>(sfpi::vConstIntPrgm0 - rsqrt_i);
                p = p * t + 4.7780996e-10f;
                sfpi::vFloat c = abs_x * rsqrt_y;
                p = p * t + 6.7723136e-08f;
                c = (-rsqrt_y) * c;
                p = p * t + 6.7822934e-06f;
                sfpi::vFloat k = sfpi::vConstFloatPrgm2 + c;
                p = p * t + 4.3402635e-04f;
                k = c * k + sfpi::vConstFloatPrgm1;
                p = p * t + 0.015625f;
                rsqrt_y = rsqrt_y * k;
                p = p * t + 0.25f;
                c = abs_x * rsqrt_y;
                const sfpi::vFloat half_y = sfpi::addexp(rsqrt_y, -1 /* exp */);
                c = (-rsqrt_y) * c + 1.0f;
                val = p * t + 1.0f;
                rsqrt_y = c * half_y + rsqrt_y;

                // Q(u), u = 1/|x| = rsqrt_y^2 in [1/88.5, 1/6]: degree 5, fit 3.4e-8
                // relative, the top four coefficients bf16- or fp16-exact. Evaluated
                // here, ahead of the |x| > 6 compare, whose two instructions then
                // separate Q's last SFPMAD from the multiply by rsqrt_y; q5 is
                // materialised before rsqrt_y^2 for the same reason.
                const sfpi::vFloat q5 = 0.62109375f;
                const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;
                q = PolynomialEvaluator::eval(
                    inv_abs_x,
                    3.9894217e-01f,
                    4.9883097e-02f,
                    0.0273284912109375f,
                    0.04400634765625f,
                    -0.0947265625f,
                    q5);
            } else {
                // Degree 5 in t, 1.3e-4 relative, every coefficient bf16- or
                // fp16-exact: bf16 output cannot see the terms the 32-bit fit
                // carries beyond it.
                sfpi::vFloat p = t * 1.2572855e-07f + 4.7385693e-06f;
                rsqrt_y = sfpi::as<sfpi::vFloat>(sfpi::vConstIntPrgm0 - rsqrt_i);
                p = p * t + 4.632473e-04f;
                sfpi::vFloat c = abs_x * rsqrt_y;
                p = p * t + 0.01546478271484375f;
                c = (-rsqrt_y) * c;
                p = p * t + 0.250244140625f;
                sfpi::vFloat k = sfpi::vConstFloatPrgm2 + c;
                val = p * t + 1.0f;
                k = c * k + sfpi::vConstFloatPrgm1;
                rsqrt_y = rsqrt_y * k;
            }
        }

        // ─── Asymptotic overwrite for |x| > 6; +inf past 88.5 ─────────────
        // Overflow and +/-inf -> +inf is nested inside the asymptotic block
        // (|x| > 88.5 implies |x| > 6), one predicate round-trip fewer per
        // iteration; NaN lanes keep the NaN that region 1's arithmetic
        // propagates, or take it from the SFPMUL. See the file-level comment.
        v_if(abs_x > I0_THRESHOLD) {
            if constexpr (is_fp32_dest_acc_en) {
                const sfpi::vFloat scale = rsqrt_y * q;
                val = _sfpu_exp_fp32_accurate_nonneg_unsafe_(abs_x) * scale;
            } else {
                val = calculate_i0_asymptotic_bf16_(abs_x, rsqrt_y);
            }
            v_if(abs_x > I0_MAX_INPUT) { val = abs_x * std::numeric_limits<float>::infinity(); }
            v_endif;
        }
        v_endif;

        // A bf16-in/float32-out call (ttnn.i0(bf16_tensor, output_tensor=<float32
        // tensor>), a supported mixed-dtype combination) runs DEST in 32-bit mode
        // with INP_FLOAT32 undefined, and its output must not be rounded down to
        // BF16 here -- hence the DEST-width key.
        if constexpr (!is_fp32_dest_acc_en) {
            val = sfpi::convert<sfpi::vFloat16b>(val, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

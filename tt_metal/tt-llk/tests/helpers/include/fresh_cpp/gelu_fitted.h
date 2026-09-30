// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CM, 2026-08-19).
// Fitted gelu vendored from the tt-polynomial-fitter frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/gelu_p6_s5_chebyshev_any_ulp.csv (BH, bf16)
//   kernel shape : tt-metal branch nkapre/tt-polynomial-fitter @ 8063ae8eced6529bd5fa9d8336066601eaa4fd67
//                  tt_metal/programming_examples/generic_lut_activation_embedded/
//                  kernels/compute/piecewise_generic.cpp — piecewise_generic_lut
//                  poly cascade (segment-0 Horner, then per-segment
//                  v_if(x >= boundary) overwrite; full-degree Horner every
//                  segment, no range reduction) + the ASYMPTOTIC_FACTOR_
//                  EXP_QUADRATIC post-apply (piecewise_generic_specialized.cpp):
//                  segment 0 is an asymptotic CORRECTION polynomial whose
//                  value is multiplied, for lanes x < -3, by the dominant
//                  factor exp(-x^2/2) * (-1/sqrt(2*pi)) computed with the
//                  kernel's Cody-Waite asymptotic_exp (deg-5 Taylor).
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-19).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners P6/s5 chebyshev):
//   max_ulp_pure_bf16 128.0, 6.93 us vs TTNN 254.93 ulp @ 5.75 us — NOTE this
//   is the frontier's ONE runtime loss (selected by the loss_rescue ML-parity
//   rule): superior ULP, INFERIOR runtime vs TTNN's native gelu.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

#include <cstdint>

namespace ckernel::sfpu
{

// Fitted gelu (frontier winner gelu_p6_s5_chebyshev): 5-segment degree-6
// polynomial cascade over [-10, 10] (chebyshev-placed interior boundaries
// -3, -1, 0.5, 2.78125).  Segment 4 is the fitter's affine identity tail
// (c2..c6 = 0); the measured kernel still runs the full-degree Horner there,
// and this body mirrors that arithmetic exactly.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_gelu_fitted_cpp()
{
    // gelu_p6_s5_chebyshev_any_ulp.csv rows 0..4: {c0..c6} per segment.
    constexpr float S0[7] = {
        4.7860594117782379e-01f,
        -3.2355804382887093e-01f,
        -9.5759156178433377e-02f,
        -1.6109949459650771e-02f,
        -1.5728624913275179e-03f,
        -8.3080477542880657e-05f,
        -1.8374107441665248e-06f};
    constexpr float S1[7] = {
        1.2485267221927643e-01f,
        9.6500647068023682e-01f,
        1.0980057716369629e+00f,
        5.2933466434478760e-01f,
        1.2681663036346436e-01f,
        1.4527644962072372e-02f,
        5.9556961059570312e-04f};
    constexpr float S2[7] = {
        0.0000000000000000e+00f,
        4.9998520910396588e-01f,
        3.9891171908897050e-01f,
        4.4619410034943516e-04f,
        -6.5795805183231623e-02f,
        -1.6761105531056179e-03f,
        6.9887704889105883e-03f};
    constexpr float S3[7] = {
        4.3814483586127320e-03f,
        4.7702547103854387e-01f,
        4.4005074217984103e-01f,
        -2.0788573676826964e-02f,
        -8.8591043507175726e-02f,
        3.2934775170905838e-02f,
        -3.6604108395746579e-03f};
    constexpr float S4[7] = {
        -1.0440399255898063e-02f,
        1.0018974177931772e+00f,
        0.0000000000000000e+00f,
        0.0000000000000000e+00f,
        0.0000000000000000e+00f,
        0.0000000000000000e+00f,
        0.0000000000000000e+00f};
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        // Segment 0 as the all-lane default, then predicated overwrites in
        // ascending boundary order (the measured cascade shape).
        sfpi::vFloat r = ((((((S0[6] * x + S0[5]) * x + S0[4]) * x + S0[3]) * x + S0[2]) * x + S0[1]) * x + S0[0]);
        v_if (x >= -3.0f)
        {
            r = ((((((S1[6] * x + S1[5]) * x + S1[4]) * x + S1[3]) * x + S1[2]) * x + S1[1]) * x + S1[0]);
        }
        v_endif;
        v_if (x >= -1.0f)
        {
            r = ((((((S2[6] * x + S2[5]) * x + S2[4]) * x + S2[3]) * x + S2[2]) * x + S2[1]) * x + S2[0]);
        }
        v_endif;
        v_if (x >= 0.5f)
        {
            r = ((((((S3[6] * x + S3[5]) * x + S3[4]) * x + S3[3]) * x + S3[2]) * x + S3[1]) * x + S3[0]);
        }
        v_endif;
        v_if (x >= 2.78125f)
        {
            r = ((((((S4[6] * x + S4[5]) * x + S4[4]) * x + S4[3]) * x + S4[2]) * x + S4[1]) * x + S4[0]);
        }
        v_endif;
        // Asymptotic post-apply (dominant factor -exp(-x^2/2)/sqrt(2*pi),
        // ASYMPTOTIC_UPPER_BOUND = -3.0 = the seg0/seg1 boundary): segment 0
        // holds a correction polynomial; multiply it by the Cody-Waite
        // exp(-x^2/2) and the -1/sqrt(2*pi) scale, exactly the measured
        // kernel's asymptotic_exp arithmetic (deg-5 Taylor, magic-number
        // round, hi/lo ln2 split, exponent recombine).
        v_if (x < -3.0f)
        {
            // arg is clamped at the point where exp(arg) underflows fp32 anyway
            // (-88 < -127*ln2), for two reasons beyond the value.  The magic-number
            // round below recovers its integer from as<vInt>(z + 1.5*2^23) -
            // as<vInt>(1.5*2^23), which is only meaningful while z + 1.5*2^23 stays
            // POSITIVE: past |x| = 4176 it goes negative and k_int is whatever the
            // integer format makes of a negative float reinterpreted, which on
            // silicon comes back with the wrong SIGN (measured: x = -4192 returned
            // +4.2573e15 with the exponent guard alone, where an int32 two's-complement
            // host model predicts 0 - so the host model cannot settle this case and the
            // clamp removes the dependence on it).  With arg >= -88, z >= -127, the
            // bias trick is inside its documented |z| < 2^22 range, and k_int >= -127.
            const sfpi::vFloat arg   = sfpi::max(x * x * -0.5f, -88.0f);
            const sfpi::vFloat z     = arg * 1.4426950408889634f;
            const sfpi::vFloat c231  = 12582912.0f; // 0x4B400000 = 1.5 * 2^23
            const sfpi::vFloat tmp   = z + c231;
            const sfpi::vFloat k     = tmp - c231;
            const sfpi::vInt k_int  = sfpi::as<sfpi::vInt>(tmp) - sfpi::as<sfpi::vInt>(c231);
            sfpi::vFloat rr         = k * -0.6931152343750000f + arg;
            rr                      = k * -3.19461832987e-05f + rr;
            sfpi::vFloat p          = 1.0f / 120.0f;
            p                       = p * rr + 1.0f / 24.0f;
            p                       = p * rr + 1.0f / 6.0f;
            p                       = p * rr + 0.5f;
            p                       = p * rr + 1.0f;
            p                       = p * rr + 1.0f;
            const sfpi::vInt pexp    = sfpi::exexp(p, sfpi::ExponentMode::Biased);
            const sfpi::vInt new_exp = pexp + k_int;
            // exp(-x^2/2) UNDERFLOWS long before x leaves this arm, and setexp
            // keeps only the low 8 bits of its exponent operand, so once
            // new_exp goes non-positive the field WRAPS and the underflowed
            // factor comes back as a NORMAL number: at x = -13.3125 new_exp is
            // -1 -> field 255 -> NaN, at -13.375 it is -3 -> -6.1144e37, at -20
            // -> +3.7471e-10 with the SIGN FLIPPED against a function that is
            // negative and bounded by -0.169979 on this whole arm.
            // Guarded exactly as the in-tree siblings guard the same
            // reconstruction -- softplus_exp_negative (v_if(new_exp > 0), FTZ
            // otherwise) and xielu's -126.5 clamp.  Flush-to-zero IS the true
            // fp32 value here: gelu(-14) = -1.09e-43, gelu(-20) = -5.5e-88.
            // NOTE this is deliberately NOT the x <= -5.54259443 flush the
            // sibling gelu.h / gelu_255_licensed.h bodies carry.  Those flush
            // because their own exp region is bracketed there; this asymptotic
            // arm is EXACT from -3 down to -13 (0 / 0 / 9 bf16 ULP at -5 /
            // -10 / -13 against an mpmath golden), and flushing at -5.5426
            // would trade a wrap defect for a 19535-to-25928 bf16 ULP band of
            // the very plausible constant 0 shape this class of defect is
            // about.  The guard binds only where setexp cannot represent the
            // result, which is the argument-side statement of the bug.
            // The result is set, not multiplied by a zeroed factor: segment 0's
            // degree-6 correction polynomial itself overflows to +/-inf past
            // |x| ~ 1.1e7, and inf * 0 is a NaN where the answer is -0.
            v_if (new_exp > 0)
            {
                r = r * sfpi::setexp(p, new_exp) * -3.9894228040143270e-01f;
            }
            v_else
            {
                r = 0.0f;
            }
            v_endif;
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

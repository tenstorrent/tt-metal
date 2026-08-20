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
            const sfpi::vFloat arg  = x * x * -0.5f;
            const sfpi::vFloat z    = arg * 1.4426950408889634f;
            const sfpi::vFloat c231 = 12582912.0f; // 0x4B400000 = 1.5 * 2^23
            const sfpi::vFloat tmp  = z + c231;
            const sfpi::vFloat k    = tmp - c231;
            const sfpi::vInt k_int  = sfpi::as<sfpi::vInt>(tmp) - sfpi::as<sfpi::vInt>(c231);
            sfpi::vFloat rr         = k * -0.6931152343750000f + arg;
            rr                      = k * -3.19461832987e-05f + rr;
            sfpi::vFloat p          = 1.0f / 120.0f;
            p                       = p * rr + 1.0f / 24.0f;
            p                       = p * rr + 1.0f / 6.0f;
            p                       = p * rr + 0.5f;
            p                       = p * rr + 1.0f;
            p                       = p * rr + 1.0f;
            const sfpi::vInt pexp   = sfpi::exexp(p, sfpi::ExponentMode::Biased);
            const sfpi::vFloat e    = sfpi::setexp(p, pexp + k_int);
            r                       = r * e * -3.9894228040143270e-01f;
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

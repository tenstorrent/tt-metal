// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CW, 2026-08-20).
// Fitted acosh over the CORPUS CONTRACT domain U[1, 10] INCLUDING the x = 1
// branch point — the lane-CR wave-2 TESTED refusal ("fit [1.01, 100]
// extrapolates 0.0579 at x = 1 where the golden is 0 > atol 0.05; fitted arm
// FAILED CRAQ on the bh sim") resolved by a branch-point-factored refit:
//   coefficients : tenstorrent/tt-polynomial-fitter branch agent/rlibm-refits
//                  @ c73c618a63393d60fd0e7fdf2330a319d44c174d
//                  data/coefficients/acosh_u10_p3_s1_uniform_rlibm_ulp_branchpt.csv
//                  (fit target activations/acosh_u10.json; is_asymptotic
//                  segment, dominant factor sqrt(x-1), correction polynomial
//                  in u = x - 1).
//   method       : scripts/rlibm_interval_fit.py — RLIBM-style rounding-interval
//                  LP (The-RLIBM-Project interval-gen + polynomial-gen/polygen.cpp
//                  rlibm_solve_with_soplex form, dense via scipy HiGHS), with the
//                  kernel's fp32 sqrt path SIMULATED PER INPUT and folded into
//                  the constraint matrix (evaluation-order-aware LP, RLIBM
//                  CGO'23): every one of the 417 bf16 inputs in [1, 10]
//                  constrains sqrt_fp32(x-1) * P(x-1) to a +/-1-ulp interval
//                  around the f64 golden; normalized LP margin 0.5390; fp32
//                  replication under BOTH mad models verified in-interval.
//   quality      : max pure bf16 ULP 0.9562 (both mad models), mean 0.353;
//                  harness contract (atol 0.05, rtol 0.05) PASSES at every
//                  bf16 point INCLUDING x = 1.0, where the result is EXACTLY
//                  0 by construction (see below).
//   NOT YET on tt-polynomial-fitter main / tt-metal main (no upstream PR as
//   of 2026-08-20).
//   Shape: acosh(x) = sqrt(u) * P3(u), u = x - 1 (exact in fp32 for bf16 x:
//   Sterbenz below 2, trivially exact above).  acosh(1+u) = sqrt(2u) *
//   (1 - u/12 + ...) is analytic in u after the sqrt factor, so a degree-3
//   correction suffices where CR's un-factored R4/4 rational could not fit
//   the vertical tangent at all.  sqrt(u) uses the classic 0x5f3759df
//   inverse-sqrt seed + 2 Newton steps (rsqrt_fitted.h's constants), then
//   s = u * y.  BRANCH-POINT EXACTNESS: the Newton product order is
//   (hu*y)*y — NOT hu*(y*y) and NOT addexp(u,-1) for hu — because at u = 0
//   the seed y is ~1.3e19 and y*y overflows to inf on the second iteration
//   ((0.5*0)*inf = NaN), while (hu*y)*y stays exactly 0; addexp(-1) on
//   u = 0 fabricates 0xFF800000.  With this order u = 0 gives finite y,
//   s = 0*y = 0, result = 0 * P(0) = 0 exactly.  Out-of-domain inputs are
//   NOT clamped (the corpus stimulus is the contract; x < 1 is a registered
//   undefined-range hole).
//   RE-SYNC: when the rlibm refits merge upstream or the fitter refits,
//   re-derive from the then-current frontier selection.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted acosh (rlibm interval-LP winner acosh_u10 P3, sqrt(x-1)-factored).
template <int ITERATIONS>
__attribute__((noinline)) void calculate_acosh_fitted_cpp()
{
    constexpr int MAGIC        = 0x5f3759df; // inverse-sqrt seed (rsqrt_fitted.h)
    constexpr float THREE_HALF = 1.5f;
    // Correction polynomial in u = x - 1, low-to-high (c0 ~= sqrt(2)).
    constexpr float C0 = 1.4111452102661133e+00f;
    constexpr float C1 = -1.0064046829938889e-01f;
    constexpr float C2 = 1.0241696611046791e-02f;
    constexpr float C3 = -4.6586585813201964e-04f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x  = sfpi::dst_reg[0];
        const sfpi::vFloat u  = x - 1.0f; // exact for bf16 x in [1, 10]
        sfpi::vFloat y        = sfpi::as<sfpi::vFloat>(sfpi::vInt(MAGIC) - sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(u) >> 1));
        const sfpi::vFloat hu = 0.5f * u; // NOT addexp(u,-1): u==0 must stay +0
        // 2 Newton steps; (hu*y)*y order keeps u == 0 exact (never 0*inf).
        y                    = y * (THREE_HALF - (hu * y) * y);
        y                    = y * (THREE_HALF - (hu * y) * y);
        const sfpi::vFloat s = u * y; // sqrt(u); exactly 0 at u == 0
        sfpi::vFloat p       = C3;
        p                    = p * u + C2;
        p                    = p * u + C1;
        p                    = p * u + C0;
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(s * p, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

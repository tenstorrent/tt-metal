// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CR, 2026-08-20).
// Fitted digamma vendored from the tt-polynomial-fitter frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/digamma_n4d4_s1_uniform_rational_ulp.csv
//                  (BH, bf16; rational_cascade R4/4, single segment on
//                  [0.01, 102], rational_rminimax).
//   kernel shape : same repo/sha deployment/generic_lut_activation/kernels/compute/
//                  piecewise_rational.cpp — eval_rational_interleaved<4,4> +
//                  sfpu_reciprocal_iter<2> (also on tt-metal branch
//                  nkapre/tt-polynomial-fitter @ 8063ae8eced6).  The measured
//                  kernel interleaves the two Horner chains for ILP; the
//                  sequential chains below are value-identical (independent
//                  chains, same fp32 op sequence per chain) — scheduling is
//                  the corpus compiler's job.
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-20).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners R4/4/s1):
//   max_ulp_pure_bf16 0.6577, 2.65 us vs TTNN 1.2663 ulp @ 6.64 us.
//   Corpus digamma contract: U[0.1, 50] (inside the fit domain), golden
//   torch.digamma.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted digamma (frontier winner digamma_n4d4_s1): psi(x) ~= N(x) / D(x),
// one rational segment over the whole domain; the divide is the production
// sfpu reciprocal restated as the corpus-blessed fresh_recip<2>.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_digamma_fitted_cpp()
{
    constexpr float N0 = -1.0017638206481934e+00f;
    constexpr float N1 = -1.3880373239517212e+00f;
    constexpr float N2 = 9.9295830726623535e-01f;
    constexpr float N3 = 2.8472983837127686e-01f;
    constexpr float N4 = 4.4142566621303558e-03f;
    constexpr float D0 = 1.1938915122300386e-05f;
    constexpr float D1 = 1.0f;
    constexpr float D2 = 8.3426612615585327e-01f;
    constexpr float D3 = 8.3190590143203735e-02f;
    constexpr float D4 = 6.8405340425670147e-04f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat num     = N4;
        num                  = num * x + N3;
        num                  = num * x + N2;
        num                  = num * x + N1;
        num                  = num * x + N0;
        sfpi::vFloat denom   = D4;
        denom                = denom * x + D3;
        denom                = denom * x + D2;
        denom                = denom * x + D1;
        denom                = denom * x + D0;
        const sfpi::vFloat y = num * fresh_recip<2>(denom);
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

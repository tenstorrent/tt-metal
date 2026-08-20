// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CR, 2026-08-20).
// Fitted polygamma (order 1, trigamma) vendored from the tt-polynomial-fitter
// frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/polygamma_n4d4_s1_uniform_rational_ulp_polish.csv
//                  (BH, bf16; rational_cascade R4/4, single segment on
//                  [0.01, 10], rational_rminimax + polish).  Verified against
//                  trigamma: N(1)/D(1) = 1.64494 = pi^2/6 = psi_1(1); the
//                  corpus golden is torch.polygamma(1, x) — same order.
//   kernel shape : same repo/sha deployment/generic_lut_activation/kernels/compute/
//                  piecewise_rational.cpp — eval_rational_interleaved<4,4> +
//                  sfpu_reciprocal_iter<2> (also on tt-metal branch
//                  nkapre/tt-polynomial-fitter @ 8063ae8eced6).  Interleaving
//                  is scheduling-only; the sequential chains below are
//                  value-identical.
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-20).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners R4/4/s1):
//   max_ulp_pure_bf16 0.4999, 2.65 us vs TTNN 0.5095 ulp @ 5.75 us.
//   Corpus polygamma contract: U[0.5, 10] (inside the fit domain).
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted polygamma order 1 (frontier winner polygamma_n4d4_s1 polish):
// psi_1(x) ~= N(x) / D(x), one rational segment; fresh_recip<2> divide.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_polygamma_fitted_cpp()
{
    constexpr float N0 = 3.9437353610992432e-01f;
    constexpr float N1 = 1.0008291006088257e+00f;
    constexpr float N2 = 1.3650107383728027e+00f;
    constexpr float N3 = 7.2213292121887207e-01f;
    constexpr float N4 = 2.2817848730483092e-05f;
    constexpr float D0 = -3.2120799353663188e-09f;
    constexpr float D1 = -5.7001284403668251e-07f;
    constexpr float D2 = 3.9441752433776855e-01f;
    constexpr float D3 = 1.0f;
    constexpr float D4 = 7.2261381149291992e-01f;
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

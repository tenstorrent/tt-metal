// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CR, 2026-08-20).
// Fitted i1 (modified Bessel I1) vendored from the tt-polynomial-fitter
// frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/i1_n9d3_s1_uniform_rational_ulp.csv
//                  (BH, bf16; rational_cascade, single segment on [-10, 10],
//                  rational_rminimax; odd numerator / even denominator — the
//                  codegen's RATIONAL_NUM_PARITY_ODD + RATIONAL_DEN_PARITY_EVEN
//                  x^2-Horner lowering, N(x) = x * H_n(x^2), D(x) = H_d(x^2).
//                  The frontier cfg string prints R9/2; the winner CSV's own
//                  rows carry num_degree 9 / den_degree 3 with d3 == 0 —
//                  transcribed from the CSV, the authority).
//   kernel shape : same repo/sha deployment/generic_lut_activation/kernels/compute/
//                  piecewise_rational.cpp — eval_rational_parity<9,3> +
//                  sfpu_reciprocal_iter<2> (also on tt-metal branch
//                  nkapre/tt-polynomial-fitter @ 8063ae8eced6).
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-20).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners R9/2/s1):
//   max_ulp_pure_bf16 0.7469, 2.12 us vs TTNN 128.0 ulp @ 7.36 us.
//   Corpus i1 contract: U[-3.75, 3.75] (inside the fit domain).
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted i1 (frontier winner i1 n9/d3 parity): I1(x) ~= x*Hn(x^2) / Hd(x^2)
// — the parity x^2-Horner halves the rung count; fresh_recip<2> divide.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_i1_fitted_cpp()
{
    constexpr float N1 = 5.0062245130538940e-01f;
    constexpr float N3 = 5.8874081820249557e-02f;
    constexpr float N5 = 2.5045361835509539e-03f;
    constexpr float N7 = 2.6910716769634746e-05f;
    constexpr float N9 = 7.1203459128810209e-07f;
    constexpr float D0 = 1.0f;
    constexpr float D2 = -5.1558478735387325e-03f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x     = sfpi::dst_reg[0];
        const sfpi::vFloat x2    = x * x;
        sfpi::vFloat num         = N9;
        num                      = num * x2 + N7;
        num                      = num * x2 + N5;
        num                      = num * x2 + N3;
        num                      = num * x2 + N1;
        num                      = num * x; // odd parity: N(x) = x * H(x^2)
        const sfpi::vFloat denom = D2 * x2 + D0;
        const sfpi::vFloat y     = num * fresh_recip<2>(denom);
        sfpi::dst_reg[0]         = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

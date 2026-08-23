// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CR, 2026-08-20).
// Fitted i0 (modified Bessel I0) vendored from the tt-polynomial-fitter
// frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/i0_n6d6_s1_uniform_rational_ulp.csv
//                  (BH, bf16; rational_cascade R6/6, single segment on
//                  [-10, 10], rational_rminimax; all odd-index coefficients
//                  are exactly zero).
//   kernel shape : same repo/sha deployment/generic_lut_activation/kernels/compute/
//                  piecewise_rational.cpp — eval_rational_interleaved<6,6> +
//                  sfpu_reciprocal_iter<2> (also on tt-metal branch
//                  nkapre/tt-polynomial-fitter @ 8063ae8eced6).  The measured
//                  kernel runs the FULL degree-6 Horner (the codegen's parity
//                  x^2 lowering only fires for odd-num/even-den shapes), so
//                  the zero-coefficient rungs below stay as bare multiplies —
//                  bit-identical to the kernel's a*x + 0 rungs.  Interleaving
//                  is scheduling-only.
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-20).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners R6/6/s1):
//   max_ulp_pure_bf16 0.5955, 2.75 us vs TTNN 1.0193 ulp @ 3.10 us.
//   Corpus i0 contract: U[-3.75, 3.75] (inside the fit domain).
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

// LANE-GC WINNER-REFRESH AUDIT (2026-08-23): still the frontier winner at
// tt-polynomial-fitter origin/main 4cbc636d7fc7202d01a8bdb6ea08eb214445e05e —
// data/coefficients/i0_n6d6_s1_uniform_rational_ulp.csv byte-identical since
// the vendoring sha; tier_silicon_summary.csv i0 row = R6/6/s1, certified
// max_ulp_pure_bf16 0.5955 @ 2.75us vs TTNN 1.0193 @ 3.10us.  Corpus-domain
// re-verification (U[-3.75, 3.75] exhaustive bf16, scipy i0 fp64 golden,
// ttpoly units.py pure-ULP semantics, two-rounding AND fma_model_bh mad
// models): max 0.5955 — golden agreement exact.  No eltwise-bodies PR filed
// upstream as of 2026-08-23 (lane GB recon).  Coefficients unchanged by
// lane GC.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted i0 (frontier winner i0_n6d6_s1): I0(x) ~= N(x) / D(x), one rational
// segment, full-degree Horner in x with the zero rungs folded (a*x + 0 == a*x
// bitwise); fresh_recip<2> divide.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_i0_fitted_cpp()
{
    constexpr float N0 = 9.9948841333389282e-01f;
    constexpr float N2 = 2.3273536562919617e-01f;
    constexpr float N4 = 1.0797064751386642e-02f;
    constexpr float N6 = 2.0466510613914579e-04f;
    constexpr float D0 = 1.0f;
    constexpr float D2 = -1.8484041094779968e-02f;
    constexpr float D4 = 1.3244326692074537e-04f;
    constexpr float D6 = -3.5643680007524381e-07f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat num     = N6;
        num                  = num * x; // n5 == 0
        num                  = num * x + N4;
        num                  = num * x; // n3 == 0
        num                  = num * x + N2;
        num                  = num * x; // n1 == 0
        num                  = num * x + N0;
        sfpi::vFloat denom   = D6;
        denom                = denom * x; // d5 == 0
        denom                = denom * x + D4;
        denom                = denom * x; // d3 == 0
        denom                = denom * x + D2;
        denom                = denom * x; // d1 == 0
        denom                = denom * x + D0;
        const sfpi::vFloat y = num * fresh_recip<2>(denom);
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

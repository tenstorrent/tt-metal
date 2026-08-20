// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CR, 2026-08-20).
// Fitted mish vendored from the tt-polynomial-fitter frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/mish_n9d4_s1_uniform_rational_ulp.csv
//                  (BH, bf16; rational_cascade R9/4, single segment on
//                  [-10, 10], rational_rminimax).
//   kernel shape : same repo/sha deployment/generic_lut_activation/kernels/compute/
//                  piecewise_rational.cpp — eval_rational_interleaved<9,4> +
//                  sfpu_reciprocal_iter<2> (also on tt-metal branch
//                  nkapre/tt-polynomial-fitter @ 8063ae8eced6).  Interleaving
//                  is scheduling-only; the sequential chains below are
//                  value-identical.
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-20).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners R9/4/s1):
//   max_ulp_pure_bf16 0.8429, 3.49 us vs TTNN 1.4365 ulp @ 4.54 us.
//   Corpus mish contract: U[-5, 5] (inside the fit domain), golden
//   x * tanh(softplus(x)).  This replaces the production composite
//   (exp/log/tanh chain) with one direct rational.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted mish (frontier winner mish_n9d4_s1): mish(x) ~= N(x) / D(x), one
// rational segment (n0 == 0 keeps mish(0) == 0 exactly); fresh_recip<2>.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_mish_fitted_cpp()
{
    constexpr float N1 = 5.9956467151641846e-01f;
    constexpr float N2 = 1.5413552522659302e-01f;
    constexpr float N3 = 1.4200618490576744e-02f;
    constexpr float N4 = 2.6611112989485264e-03f;
    constexpr float N5 = 7.3920952854678035e-04f;
    constexpr float N6 = 8.9643493993207812e-05f;
    constexpr float N7 = 3.8300422602333128e-06f;
    constexpr float N8 = -5.1095558006863939e-08f;
    constexpr float N9 = -5.8142703984742639e-09f;
    constexpr float D0 = 1.0f;
    constexpr float D1 = -2.7285581827163696e-01f;
    constexpr float D2 = 1.9487699866294861e-01f;
    constexpr float D3 = -3.4017711877822876e-02f;
    constexpr float D4 = 4.1533689945936203e-03f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat num     = N9;
        num                  = num * x + N8;
        num                  = num * x + N7;
        num                  = num * x + N6;
        num                  = num * x + N5;
        num                  = num * x + N4;
        num                  = num * x + N3;
        num                  = num * x + N2;
        num                  = num * x + N1;
        num                  = num * x; // n0 == 0
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

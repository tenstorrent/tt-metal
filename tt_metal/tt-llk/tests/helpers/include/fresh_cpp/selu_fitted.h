// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CR, 2026-08-20).
// Fitted selu vendored from the tt-polynomial-fitter frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/selu_p11_s2_chebyshev_any_ulp.csv (BH,
//                  bf16; poly_cascade P11, two chebyshev segments on
//                  [-10, 10], boundary 0.0; segment 1 is the exact linear
//                  scale*x).  Segment 0 fits scale*alpha*(e^x - 1) with the
//                  standard selu constants (c1 = 1.7580897808 = scale*alpha;
//                  the production dispatch parks the same scale ~1.0507 /
//                  alpha ~1.6733).
//   kernel shape : same repo/sha deployment/generic_lut_activation/kernels/compute/
//                  piecewise_generic.cpp — piecewise_generic_lut segment
//                  cascade (also on tt-metal branch
//                  nkapre/tt-polynomial-fitter @ 8063ae8eced6).
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-20).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners P11/s2):
//   max_ulp_pure_bf16 0.4996, 2.47 us vs TTNN 0.4996 ulp @ 3.62 us.
//   Corpus selu contract: U[-5, 5] (inside the fit domain).  This replaces
//   the production exp-composite with one direct polynomial branch.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted selu (frontier winner selu_p11_s2): selu(x) = P11(x) for x < 0
// (c0 == 0 keeps selu(0) == 0 exactly), scale*x for x >= 0.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_selu_fitted_cpp()
{
    constexpr float C1    = 1.7580897808074951e+00f;
    constexpr float C2    = 8.7886452674865723e-01f;
    constexpr float C3    = 2.9237365722656250e-01f;
    constexpr float C4    = 7.2331503033638000e-02f;
    constexpr float C5    = 1.3937523588538170e-02f;
    constexpr float C6    = 2.1037068217992783e-03f;
    constexpr float C7    = 2.4284230312332511e-04f;
    constexpr float C8    = 2.0467858121264726e-05f;
    constexpr float C9    = 1.1732893199223327e-06f;
    constexpr float C10   = 4.0517328159239696e-08f;
    constexpr float C11   = 6.3206162526086018e-10f;
    constexpr float SCALE = 1.0507010221481323e+00f; // segment 1: y = scale*x
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat p       = C11;
        p                    = p * x + C10;
        p                    = p * x + C9;
        p                    = p * x + C8;
        p                    = p * x + C7;
        p                    = p * x + C6;
        p                    = p * x + C5;
        p                    = p * x + C4;
        p                    = p * x + C3;
        p                    = p * x + C2;
        p                    = p * x + C1;
        p                    = p * x; // c0 == 0
        v_if (x >= 0.0f)
        {
            p = SCALE * x; // segment 1: exact linear
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(p, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

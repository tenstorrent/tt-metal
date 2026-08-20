// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CR, 2026-08-20).
// Fitted elu (alpha = 1.0, the corpus contract's alpha) vendored from the
// tt-polynomial-fitter frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/elu_p15_s2_chebyshev_any_ulp.csv (BH,
//                  bf16; poly_cascade P15, two chebyshev segments on
//                  [-10, 10], boundary 0.0; segment 1 is the exact identity).
//                  Segment 0's c15 is exactly 0.0 — the coefficient set IS
//                  the celu P14 series (alpha = 1 makes the two functions
//                  identical), and the codegen's adaptive segment degree
//                  elides the zero top rung; the P14 Horner below is that
//                  emitted form.
//   kernel shape : same repo/sha deployment/generic_lut_activation/kernels/compute/
//                  piecewise_generic.cpp — piecewise_generic_lut segment
//                  cascade (also on tt-metal branch
//                  nkapre/tt-polynomial-fitter @ 8063ae8eced6).
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-20).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners P15/s2):
//   max_ulp_pure_bf16 0.4998, 2.66 us vs TTNN 0.4998 ulp @ 3.48 us.
//   Corpus elu contract: U[-5, 5] (inside the fit domain), alpha = 1.0 both
//   here and in the production dispatch scalars.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted elu (frontier winner elu_p15_s2): elu(x) = P(x) for x < 0 (a minimax
// fit of e^x - 1 on [-10, 0], c0 == 0 keeps elu(0) == 0 exactly), identity
// for x >= 0.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_elu_fitted_cpp()
{
    constexpr float C1  = 1.0f;
    constexpr float C2  = 4.9999934434890747e-01f;
    constexpr float C3  = 1.6666224598884583e-01f;
    constexpr float C4  = 4.1655816137790680e-02f;
    constexpr float C5  = 8.3194561302661896e-03f;
    constexpr float C6  = 1.3780959416180849e-03f;
    constexpr float C7  = 1.9285458256490529e-04f;
    constexpr float C8  = 2.2803982574259862e-05f;
    constexpr float C9  = 2.2368417376128491e-06f;
    constexpr float C10 = 1.7566813426128647e-07f;
    constexpr float C11 = 1.0487319457297417e-08f;
    constexpr float C12 = 4.4169295998486291e-10f;
    constexpr float C13 = 1.1582759057437997e-11f;
    constexpr float C14 = 1.4128406560995344e-13f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat p       = C14;
        p                    = p * x + C13;
        p                    = p * x + C12;
        p                    = p * x + C11;
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
            p = x; // segment 1: exact identity
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(p, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

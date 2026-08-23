// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane GC refresh, 2026-08-23;
// supersedes lane CR's 8-segment vendoring of 2026-08-20).
// Fitted lgamma vendored from the tt-polynomial-fitter CURRENT installed fit:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 4cbc636d7fc7202d01a8bdb6ea08eb214445e05e
//                  data/coefficients/lgamma_n3d3_s4_curvature_any_ulp.csv
//                  (hardwaremode bh, bf16; rational R3/3, FOUR curvature-placed
//                  segments on [0.001, 100], rational_rminimax rows).
//   spec         : activations/lgamma.json @ same sha — the 2026-08-22
//                  breakpoint-audit campaign ("only 3 breakpoints ever
//                  mattered", commit 894d21a41) cut the installed fit 8 -> 6
//                  -> 4 segments; boundaries 0.1 / 0.8 / 2.5 are each
//                  measured load-bearing (0.1 = log singularity, 0.8 / 2.5
//                  bracket the interior roots at x=1 and x=2).
//   kernel shape : deployment/generic_lut_activation/kernels/compute/
//                  piecewise_rational.cpp — the deferred-reciprocal segment
//                  cascade (segment 0 unconditional, each higher segment
//                  re-evaluates under v_if(x >= lo), ONE reciprocal at the
//                  end).  Structure unchanged from CR's vendoring; only the
//                  segment count and rows moved.
//   NOT YET on tt-metal main (lane GB recon 2026-08-23: no eltwise-bodies PR
//   filed; tier_silicon_summary.csv BH row still lists the older R3/3/s8 —
//   the s4 fit is newer than the BH frontier board and was runtime-validated
//   on WH silicon: 0.792x -> 1.8585x vs TTNN, commit 894d21a41).
//   Certified accuracy (CSV metadata, input-exhaustive bf16 on [0.001,100]):
//   max_ulp_pure_bf16 0.5888.  Lane GC corpus-domain re-verification
//   (U[1.0, 15.0] exhaustive bf16, torch.lgamma fp64 golden, ttpoly
//   units.py pure-ULP semantics, both mad models): max 0.5888 — golden
//   agreement exact; the old s8 rows measured 0.4999 on the same domain, so
//   this refresh trades +0.09 ulp (still sub-ULP) for half the cascade.
//   Corpus lgamma contract: U[1.0, 15.0] (inside the fit domain), golden
//   torch.lgamma.  This replaces the production Stirling kernel.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current activations/lgamma.json
//   installed fit + frontier summaries.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted lgamma (current installed fit lgamma_n3d3_s4 curvature): ln|Gamma(x)|
// ~= N_s(x) / D_s(x) on 4 curvature-placed segments; deferred fresh_recip<2>.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_lgamma_fitted_cpp()
{
    // Per-segment lower boundaries (segment 0 covers everything below b1).
    constexpr float B[3] = {0.1f, 0.8f, 2.5f};
    // Per-segment rational coefficients n0..n3 / d0..d3 (CSV rows 0..3).
    constexpr float N[4][4] = {
        {1.2786000297637656e-05f, 1.0139533318579197e-02f, 5.1266646385192871e-01f, 4.8732295632362366e-02f},
        {2.0907002687454224e-01f, 1.5791414976119995e+00f, -2.9526309967041016e+00f, 1.1660702228546143e+00f},
        {7.2877955436706543e-01f, -1.0607945919036865e-01f, -1.1162450313568115e+00f, 4.9354493618011475e-01f},
        {8.7764692306518555e-01f, -1.5004484653472900e+00f, 4.6406239271163940e-01f, 3.2392941415309906e-02f}};
    constexpr float D[4][4] = {
        {1.4570325674867490e-06f, 1.7787708202376962e-03f, 1.5731981396675110e-01f, 1.0f},
        {4.4391792267560959e-02f, 9.6466886997222900e-01f, 1.0f, -5.3557038307189941e-01f},
        {2.3903881013393402e-01f, 1.0f, 2.5518828630447388e-01f, -7.8967083245515823e-03f},
        {1.0f, 3.3052915334701538e-01f, 7.5287288054823875e-03f, -6.6814600359066390e-06f}};
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat num     = ((N[0][3] * x + N[0][2]) * x + N[0][1]) * x + N[0][0];
        sfpi::vFloat denom   = ((D[0][3] * x + D[0][2]) * x + D[0][1]) * x + D[0][0];
#pragma GCC unroll 3
        for (int s = 1; s < 4; ++s)
        {
            v_if (x >= B[s - 1])
            {
                num   = ((N[s][3] * x + N[s][2]) * x + N[s][1]) * x + N[s][0];
                denom = ((D[s][3] * x + D[s][2]) * x + D[s][1]) * x + D[s][0];
            }
            v_endif;
        }
        // ONE deferred reciprocal, as the measured cascade does.
        const sfpi::vFloat y = num * fresh_recip<2>(denom);
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

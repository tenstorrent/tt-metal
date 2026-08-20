// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CR, 2026-08-20).
// Fitted lgamma vendored from the tt-polynomial-fitter frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/lgamma_n3d3_s8_curvature_rational_ulp.csv
//                  (BH, bf16; rational_cascade R3/3, EIGHT curvature-placed
//                  segments on [0.001, 100], rational_rminimax).
//   kernel shape : same repo/sha deployment/generic_lut_activation/kernels/compute/
//                  piecewise_rational.cpp — the deferred-reciprocal segment
//                  cascade (unroll_segment_rational_deferred): segment 0
//                  evaluates unconditionally, each higher segment re-evaluates
//                  under v_if(x >= lo), then ONE reciprocal at the end (also
//                  on tt-metal branch nkapre/tt-polynomial-fitter @
//                  8063ae8eced6).  Interleaving within a segment is
//                  scheduling-only.
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-20).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners R3/3/s8):
//   max_ulp_pure_bf16 0.5361, 10.56 us vs TTNN 164.17 ulp @ 13.62 us.
//   Corpus lgamma contract: U[1.0, 15.0] (inside the fit domain), golden
//   torch.lgamma.  This replaces the production Stirling kernel.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted lgamma (frontier winner lgamma_n3d3_s8 curvature): ln|Gamma(x)| ~=
// N_s(x) / D_s(x) on 8 curvature-placed segments; deferred fresh_recip<2>.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_lgamma_fitted_cpp()
{
    // Per-segment lower boundaries (segment 0 covers everything below b1).
    constexpr float B[7] = {0.1f, 0.4f, 0.8f, 1.5f, 2.5f, 4.0f, 10.0f};
    // Per-segment rational coefficients n0..n3 / d0..d3 (CSV rows 0..7).
    constexpr float N[8][4] = {
        {1.2786000297637656e-05f, 1.0139533318579197e-02f, 5.1266646385192871e-01f, 4.8732295632362366e-02f},
        {-4.4133082032203674e-02f, -4.2602032423019409e-01f, 1.6373813152313232e+00f, -1.2152585983276367e+00f},
        {3.5934329032897949e-01f, 9.9142956733703613e-01f, -2.1311731338500977e+00f, 7.8042435646057129e-01f},
        {5.8873653411865234e-01f, 2.5665462017059326e-01f, -1.4153177738189697e+00f, 5.6992661952972412e-01f},
        {9.6958971023559570e-01f, -6.3953995704650879e-01f, -7.3768329620361328e-01f, 4.0752792358398438e-01f},
        {1.4372701644897461e+00f, -1.5735676288604736e+00f, -1.6494925320148468e-01f, 2.9620590806007385e-01f},
        {1.3011600971221924e+00f, -1.8847160339355469e+00f, 3.9904689788818359e-01f, 1.0815905779600143e-01f},
        {-1.2485413551330566e+00f, -2.0344480872154236e-01f, 2.6019430160522461e-01f, 5.3864102810621262e-03f}};
    constexpr float D[8][4] = {
        {1.4570325674867490e-06f, 1.7787708202376962e-03f, 1.5731981396675110e-01f, 1.0f},
        {-8.8862255215644836e-03f, -2.4083751440048218e-01f, 1.9639156758785248e-02f, 1.0f},
        {9.0025901794433594e-02f, 1.0f, 6.2632441520690918e-01f, -1.0502161830663681e-01f},
        {1.7712265253067017e-01f, 1.0f, 3.3702093362808228e-01f, -1.6954232007265091e-02f},
        {3.6696264147758484e-01f, 1.0f, 1.8438176810741425e-01f, -3.6861724220216274e-03f},
        {6.8863499164581299e-01f, 1.0f, 1.1118378490209579e-01f, -1.1105533922091126e-03f},
        {1.0f, 6.0938894748687744e-01f, 3.2225836068391800e-02f, -1.2123033229727298e-04f},
        {1.0f, 1.0897970199584961e-01f, 1.0746838524937630e-03f, -4.6176597834346467e-07f}};
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat num     = ((N[0][3] * x + N[0][2]) * x + N[0][1]) * x + N[0][0];
        sfpi::vFloat denom   = ((D[0][3] * x + D[0][2]) * x + D[0][1]) * x + D[0][0];
#pragma GCC unroll 7
        for (int s = 1; s < 8; ++s)
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

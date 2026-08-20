// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CW, 2026-08-20).
// Fitted expm1 over the CORPUS CONTRACT domain U[-5, 5] — the lane-CR wave-2
// honest-out "FIT DOMAIN NARROWER THAN CONTRACT: winner P7/s1 fits [-1, 1]"
// resolved by a domain refit:
//   coefficients : tenstorrent/tt-polynomial-fitter branch agent/rlibm-refits
//                  @ c73c618a63393d60fd0e7fdf2330a319d44c174d
//                  data/coefficients/expm1_u5_p9_s1_uniform_rlibm_ulp_rootpin.csv
//                  (fit target activations/expm1_u5.json).
//   method       : scripts/rlibm_interval_fit.py — RLIBM-style rounding-interval
//                  LP (The-RLIBM-Project interval-gen + polynomial-gen/polygen.cpp
//                  rlibm_solve_with_soplex form, solved dense via scipy HiGHS;
//                  RLIBM-32 PLDI'21 / RLIBM-ALL POPL'22 / CGO'23): every one of
//                  the 33,090 bf16 inputs in [-5, 5] constrains the fp32 result
//                  to a +/-0.5-ulp interval around the f64 golden; normalized LP
//                  margin 0.4232; fp32 Horner replication under BOTH mad models
//                  (fused / two-rounding) verified in-interval on all points.
//   quality      : max pure bf16 ULP 0.7852 (fused) / 0.7819 (two-rounding),
//                  mean 0.0277; harness contract (atol 0.05, rtol 0.05) PASSES
//                  at every bf16 point in the domain.
//   NOT YET on tt-polynomial-fitter main / tt-metal main (no upstream PR as
//   of 2026-08-20).
//   Shape: ROOTPIN — expm1(x) = x * q(x) written as a degree-9 polynomial
//   with c0 == 0 (Horner ends with a bare * x), so expm1(+/-0) == +/-0
//   exactly and relative accuracy holds through the zero crossing WITHOUT
//   range reduction.  9 slots of MAD + 1 multiply: far under the 32-slot
//   replay cliff that the production expm1 path exceeds (laneCF: 43->38
//   words, still > 32).  Out-of-domain inputs are NOT clamped (the corpus
//   stimulus is the contract).
//   RE-SYNC: when the rlibm refits merge upstream or the fitter refits,
//   re-derive from the then-current frontier selection.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted expm1 (rlibm interval-LP winner expm1_u5 P9 rootpin).
template <int ITERATIONS>
__attribute__((noinline)) void calculate_expm1_fitted_cpp()
{
    // c0 == 0 (rootpin); c1..c9 low-to-high from the coefficient CSV.
    constexpr float C1 = 9.9902772903442383e-01f;
    constexpr float C2 = 5.0072366001598862e-01f;
    constexpr float C3 = 1.6914848983287811e-01f;
    constexpr float C4 = 4.1955754160881042e-02f;
    constexpr float C5 = 7.6604215428233147e-03f;
    constexpr float C6 = 1.2336070649325848e-03f;
    constexpr float C7 = 2.4210993433371186e-04f;
    constexpr float C8 = 3.8527090509887785e-05f;
    constexpr float C9 = 2.5944186752510165e-06f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat p       = C9;
        p                    = p * x + C8;
        p                    = p * x + C7;
        p                    = p * x + C6;
        p                    = p * x + C5;
        p                    = p * x + C4;
        p                    = p * x + C3;
        p                    = p * x + C2;
        p                    = p * x + C1;
        p                    = p * x; // c0 == 0: exact at the origin
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(p, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

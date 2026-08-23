// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CW, 2026-08-20).
// Fitted threshold at the CORPUS CONTRACT parameters threshold = 5.0,
// value = 10.0 (the sfpu_operations.h dispatch scalars) — the lane-CR wave-2
// honest-out "PARAM MISMATCH: fitter fit 0/0 == relu" resolved by a
// parameter refit:
//   coefficients : tenstorrent/tt-polynomial-fitter branch agent/rlibm-refits
//                  @ c73c618a63393d60fd0e7fdf2330a319d44c174d
//                  data/coefficients/threshold_t5_v10_p1_s2_breakpoints_rlibm_ulp.csv
//                  (exact 2-segment piecewise: constant 10.0 below the
//                  boundary, identity above; no LP needed — the refit is
//                  exact algebra).  Fit target activations/threshold_t5_v10.json.
//   method       : scripts/rlibm_interval_fit.py check_threshold() — verified
//                  EXACT (max error 0.0, 0.0 pure bf16 ULP) over every one of
//                  the 33,346 bf16 values in the fit domain [-10, 10]
//                  (superset of the corpus stimulus U[-5, 5]).
//   NOT YET on tt-polynomial-fitter main / tt-metal main (no upstream PR as
//   of 2026-08-20).
//   BOUNDARY OWNERSHIP (the load-bearing detail): torch.nn.functional.threshold
//   keeps x only for STRICT x > threshold, so x == 5.0 must produce 10.0 —
//   and bf16 stimuli from U[-5, 5] DO hit 5.0 exactly.  The generic fitter
//   cascade's `x >= boundary` select would be wrong here; this kernel uses
//   the strict v_if (x > 5.0f).  NaN lanes also fall to 10.0, matching torch.
//   RE-SYNC: when the rlibm refits merge upstream or the fitter refits,
//   re-derive from the then-current frontier selection.

// LANE-GC WINNER-REFRESH AUDIT (2026-08-23): KEPT — the fitter's main-branch
// threshold winner at origin/main 4cbc636d7fc7202d01a8bdb6ea08eb214445e05e is
// still data/coefficients/threshold_p1_s32_chebyshev_any_ulp.csv, a fit of
// the threshold(0, 0) == relu spec (the lane-CR PARAM MISMATCH stands; lane
// CW's threshold_t5_v10 refit branch agent/rlibm-refits remains unmerged
// upstream).  Both this arm and the fitter winner are EXACT on their own
// contracts (tier summary: 0.0 vs 0.0 pure ULP).  Corpus-domain
// re-verification (U[-5, 5] exhaustive bf16, torch strict-> golden, ttpoly
// units.py pure-ULP semantics): max 0.0 — golden agreement exact, boundary
// x == 5.0 -> 10.0 ownership re-confirmed.  Coefficients unchanged by
// lane GC.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted threshold(x; t=5, v=10): x if x > 5.0 else 10.0 (exact).
template <int ITERATIONS>
__attribute__((noinline)) void calculate_threshold_fitted_cpp()
{
    constexpr float THRESHOLD = 5.0f;  // target_parameters threshold
    constexpr float VALUE     = 10.0f; // target_parameters value
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat p       = VALUE;
        v_if (x > THRESHOLD)
        {
            p = x; // segment 1: exact identity (strict >, torch semantics)
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(p, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CR, 2026-08-20).
// Fitted log vendored from the tt-polynomial-fitter frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/log_p5_s1_uniform_any_ulp_rootpin.csv (BH,
//                  bf16; expalu_kind=log2, basis m_minus_1 root-pinned c0==0,
//                  log_scale=ln 2, reduced mantissa domain [1,2)).
//   kernel shape : same repo/sha deployment/generic_lut_activation/kernels/compute/
//                  piecewise_generic.cpp — log_hw_eval<5> with EXPONENT_ALU_LOG2 +
//                  LOG_HW_BASIS_M_MINUS_1 (the CC-free sign-magnitude exponent
//                  conversion is the preloaded variant's idiom, bit-identical to
//                  the predicated form for every |e| <= 254; also on tt-metal
//                  branch nkapre/tt-polynomial-fitter @ 8063ae8eced6).
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-20).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners P5/s1 rootpin):
//   max_ulp_pure_bf16 0.5050, 1.70 us vs TTNN 0.7803 ulp @ 2.42 us.
//   The IEEE specials guards (log(0) = -inf, log(neg) = NaN) are SKIPPED, as in
//   the measured range-controlled config: the corpus log contract is
//   log-uniform on [1e-4, 1e3] with the golden's (-inf, 1e-6) hole excluded,
//   so the decompose input is always a positive normal.
//
// LANE-GC WINNER-REFRESH AUDIT (2026-08-23): P5 rootpin KEPT DELIBERATELY.
// At tt-polynomial-fitter origin/main 4cbc636d7fc7202d01a8bdb6ea08eb214445e05e
// the CSV is byte-identical and P5/s1 rootpin is STILL the BH silicon board's
// selected headline winner (pareto_winners.csv selected=1, headline gate
// pure<=ttnn_pure at lowest runtime).  The tier_silicon_summary.csv log row
// additionally carries a cr_uplift tier overlay (pure<=0.5 within a 5%
// runtime budget) selecting log_p6_s1_uniform_any_ulp_rootpin_polish
// (0.4996 @ 1.77us vs P5's 0.5050 @ 1.70us) — lane GC vendored that P6 arm
// and MEASURED it on our vehicle at ON-28/pin-26: KERNEL vs-hand +39.81%
// (103864 cy) against P5's booked +23.27% (91576 cy) — the extra Horner rung
// costs ~13% here; run headline-laneGC-fitted-20260823 (corr device-golden
// PASS, CRAQ pinned-sim PASS both legs).  Per the honest-booking rule the
// FASTER arm stays live; the accuracy delta (0.5050 vs 0.4996 pure ULP) is
// contract-irrelevant on our board (both sub-ULP, gates identical).
// Corpus-domain re-verification of THIS P5 arm (log-uniform contract
// [1e-4, 1e3] exhaustive bf16, np.log fp64 golden, ttpoly units.py pure-ULP
// semantics, two-rounding AND fma_model_bh mad models): max 0.5050 — golden
// agreement exact.  No eltwise-bodies PR filed upstream as of 2026-08-23
// (lane GB recon).
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted log (frontier winner log_p5_s1 rootpin): ln(x) = (e + log2(m)) * ln 2
// for x = 2^e * m, m in [1,2) — exponent free via exexp, mantissa normalized
// via setexp, degree-5 Horner in u = m - 1 (root-pinned: c0 == 0 makes
// h(1) == 0 exactly), then one scale multiply.  Mirrors log_hw_eval<5>.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_log_fitted_cpp()
{
    // log_p5_s1_uniform_any_ulp_rootpin.csv segment 0: h(u) = log2(1+u), u in [0,1).
    constexpr float C1 = 1.4426950216293335e+00f;
    constexpr float C2 = -7.1393853425979614e-01f;
    constexpr float C3 = 4.2489251494407654e-01f;
    constexpr float C4 = -2.0009706914424896e-01f;
    constexpr float C5 = 4.6448059380054474e-02f;
    // expalu_log_scale: ln 2.
    constexpr float SCALE = 6.9314718055994529e-01f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        // Decompose x = 2^e * m, m in [1,2).
        const sfpi::vInt e_int = sfpi::exexp(x);
        const sfpi::vFloat m   = sfpi::setexp(x, 127);
        const sfpi::vFloat u   = m - 1.0f;
        // CC-free sign-magnitude conversion (int32_to_float wants
        // sign-magnitude; the corpus digamma body's proven idiom).
        const sfpi::vFloat e_float = sfpi::convert<sfpi::vFloat>(sfpi::convert<sfpi::vSMag>(e_int), sfpi::RoundMode::Nearest);
        // Degree-5 Horner for log2(m) in the root-pinned (m-1) basis (c0 == 0).
        sfpi::vFloat h       = C5;
        h                    = h * u + C4;
        h                    = h * u + C3;
        h                    = h * u + C2;
        h                    = h * u + C1;
        h                    = h * u;
        const sfpi::vFloat y = (e_float + h) * SCALE;
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

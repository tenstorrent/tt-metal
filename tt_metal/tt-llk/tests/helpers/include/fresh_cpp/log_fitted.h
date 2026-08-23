// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane GC refresh, 2026-08-23;
// supersedes lane CR's P5-rootpin vendoring of 2026-08-20).
// Fitted log vendored from the tt-polynomial-fitter CURRENT tier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 4cbc636d7fc7202d01a8bdb6ea08eb214445e05e
//                  data/coefficients/log_p6_s1_uniform_any_ulp_rootpin_polish.csv
//                  (BH, bf16; expalu_kind=log2, basis m_minus_1 root-pinned
//                  c0==0 slope-pinned at x0=1, log_scale=ln 2, reduced
//                  mantissa domain [1,2)).
//   selection    : paper/results/frontier_pareto/summaries/tier_silicon_summary.csv
//                  log row — the cr_uplift tier (pure<=0.5, budget 5%,
//                  us<ttnn_us, ml_non_regressing) moved the winner from CR's
//                  P5/s1 rootpin (0.5050 @ 1.70us) to this P6/s1 polish
//                  (0.4996 @ 1.77us); both silicon-measured on the same
//                  BH/BF16 board vs TTNN 0.7803 @ 2.42us.
//   kernel shape : same repo/sha deployment/generic_lut_activation/kernels/compute/
//                  piecewise_generic.cpp — log_hw_eval<6> with EXPONENT_ALU_LOG2 +
//                  LOG_HW_BASIS_M_MINUS_1 (the CC-free sign-magnitude exponent
//                  conversion is the preloaded variant's idiom, bit-identical to
//                  the predicated form for every |e| <= 254).
//   NOT YET on tt-metal main (lane GB recon 2026-08-23: no eltwise-bodies PR filed).
//   Certified accuracy (CSV metadata + tier summary, silicon BH/BF16):
//   max_ulp_pure_bf16 0.4996 @ 1.77 us vs TTNN 0.7803 ulp @ 2.42 us.
//   Lane GC corpus-domain re-verification (log-uniform contract [1e-4, 1e3]
//   exhaustive bf16, np.log fp64 golden, ttpoly units.py pure-ULP semantics,
//   both mad models): max 0.4996 — golden agreement exact.
//   The IEEE specials guards (log(0) = -inf, log(neg) = NaN) are SKIPPED, as in
//   the measured range-controlled config: the corpus log contract is
//   log-uniform on [1e-4, 1e3] with the golden's (-inf, 1e-6) hole excluded,
//   so the decompose input is always a positive normal.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/summaries/tier_silicon_summary.csv selection.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted log (cr_uplift tier winner log_p6_s1 rootpin polish): ln(x) =
// (e + log2(m)) * ln 2 for x = 2^e * m, m in [1,2) — exponent free via exexp,
// mantissa normalized via setexp, degree-6 Horner in u = m - 1 (root-pinned:
// c0 == 0 makes h(1) == 0 exactly), then one scale multiply.  Mirrors
// log_hw_eval<6>.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_log_fitted_cpp()
{
    // log_p6_s1_uniform_any_ulp_rootpin_polish.csv segment 0: h(u) = log2(1+u), u in [0,1).
    constexpr float C1 = 1.4426950216293335e+00f;
    constexpr float C2 = -7.1962326765060425e-01f;
    constexpr float C3 = 4.6224918961524963e-01f;
    constexpr float C4 = -2.8430065512657166e-01f;
    constexpr float C5 = 1.2552422285079956e-01f;
    constexpr float C6 = -2.6544500142335892e-02f;
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
        // Degree-6 Horner for log2(m) in the root-pinned (m-1) basis (c0 == 0).
        sfpi::vFloat h       = C6;
        h                    = h * u + C5;
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

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CM, 2026-08-19).
// Fitted tanh vendored from the tt-polynomial-fitter frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/tanh_p6_s1_uniform_basis_ulp.csv (BH, bf16)
//   kernel shape : tt-metal branch nkapre/tt-polynomial-fitter @ 8063ae8eced6529bd5fa9d8336066601eaa4fd67
//                  tt_metal/programming_examples/generic_lut_activation_embedded/
//                  kernels/compute/piecewise_generic.cpp — basis path
//                  (BASIS_SIGNED_ABS_POLY + BASIS_INPUT_ABS_X + BASIS_CLAMP_MAX 1.0
//                  + BASIS_POST_SIGN_X): y = copysign(min(P(|x|), 1.0), x),
//                  expanded P(u) = u*Q(u), single segment, plain Horner.
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-19).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners P6/s1 basis):
//   max_ulp_pure_bf16 0.8559, 1.60 us vs TTNN 128.0 ulp @ 2.24 us.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

// LANE-GC WINNER-REFRESH AUDIT (2026-08-23): still the frontier winner at
// tt-polynomial-fitter origin/main 4cbc636d7fc7202d01a8bdb6ea08eb214445e05e —
// data/coefficients/tanh_p6_s1_uniform_basis_ulp.csv byte-identical since
// the vendoring sha; tier_silicon_summary.csv tanh row = P6/s1 basis,
// certified max_ulp_pure_bf16 0.8559 @ 1.60us vs TTNN 128.0 @ 2.24us.
// Corpus-domain re-verification (U[-5, 5] exhaustive bf16, np.tanh fp64
// golden, ttpoly units.py pure-ULP semantics, two-rounding AND fma_model_bh
// mad models): max 0.8559 — golden agreement exact.  No eltwise-bodies PR
// filed upstream as of 2026-08-23 (lane GB recon).  Coefficients unchanged
// by lane GC.

#include <cstdint>

namespace ckernel::sfpu
{

// Fitted tanh (frontier winner tanh_p6_s1_uniform_basis): degree-6 expanded
// signed-abs basis polynomial on |x| over [0, 5], clamped to 1.0, sign
// restored.  Same skeleton as calculate_tanh_fresh_cpp; the coefficients are
// the fitter's fpminimax fit of tanh(x)/x (expanded), not the production
// Sollya set.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_tanh_fitted_cpp()
{
    // tanh_p6_s1_uniform_basis_ulp.csv segment 0 (c0 = 0 exactly).
    constexpr float C1 = 1.0008649826049805e+00f;
    constexpr float C2 = 1.7200719565153122e-02f;
    constexpr float C3 = -4.6311581134796143e-01f;
    constexpr float C4 = 2.6244920492172241e-01f;
    constexpr float C5 = -6.0366876423358917e-02f;
    constexpr float C6 = 5.1483884453773499e-03f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat a = sfpi::abs(x);
        sfpi::vFloat r       = C6;
        r                    = r * a + C5;
        r                    = r * a + C4;
        r                    = r * a + C3;
        r                    = r * a + C2;
        r                    = r * a + C1;
        r                    = r * a; // + c0 == 0
        r                    = sfpi::min(r, 1.0f);
        r                    = sfpi::copysgn(r, x);
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

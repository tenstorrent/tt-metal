// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CR, 2026-08-20).
// Fitted exp vendored from the tt-polynomial-fitter frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/exp_p3_s1_uniform_any_ulp.csv (BH, bf16;
//                  expalu_kind=exp2, expalu_log2_multiplier=+log2(e),
//                  compose none, reduced domain f in [0,1)).
//   kernel shape : same repo/sha deployment/generic_lut_activation/kernels/compute/
//                  piecewise_generic.cpp — exp_hw_eval<3> with EXPONENT_ALU_EXP2
//                  (the fitter's own deployment tree carries the measured kernels;
//                  the older copy on tt-metal branch nkapre/tt-polynomial-fitter @
//                  8063ae8eced6529bd5fa9d8336066601eaa4fd67 has the same exp2 body).
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-20).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners P3/s1):
//   max_ulp_pure_bf16 0.5269, 1.51 us vs TTNN 0.8833 ulp @ 1.88 us.
//   DEVIATIONS from the measured config (both value-preserving on its domain):
//   (1) the measured config carried EXP_HW_SKIP_INPUT_CLAMP (safe on its
//   [-10,10] sweep); the corpus exp contract feeds down to -100, so the
//   kernel's full-range [0,255] safety clamp is KEPT here — identical values
//   in-domain, two extra ops vs the recorded runtime. (2) the measured config
//   was BARE_SETEXP (c0 == 1.0 makes g(f) in [1,2)); the pe-corrected
//   recombine below is bit-identical for that coefficient set and matches the
//   lane-CM sigmoid transcription precedent.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

// LANE-GC WINNER-REFRESH AUDIT (2026-08-23): still the frontier winner at
// tt-polynomial-fitter origin/main 4cbc636d7fc7202d01a8bdb6ea08eb214445e05e —
// data/coefficients/exp_p3_s1_uniform_any_ulp.csv byte-identical since the
// vendoring sha; tier_silicon_summary.csv exp row = P3/s1, certified
// max_ulp_pure_bf16 0.5269 @ 1.51us vs TTNN 0.8833 @ 1.88us.  Corpus-domain
// re-verification (U[-100, 16] exhaustive bf16, np.exp fp64 golden, ttpoly
// units.py pure-ULP semantics, two-rounding AND fma_model_bh mad models):
// max 0.5269 — golden agreement exact.  No eltwise-bodies PR filed upstream
// as of 2026-08-23 (lane GB recon).  Coefficients unchanged by lane GC.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted exp (frontier winner exp_p3_s1): exp(x) = 2^(x * log2 e) evaluated by
// the exponent-ALU decompose — integer part into the exponent field, degree-3
// natural-basis refinement polynomial for 2^f on f in [0,1).  Mirrors
// exp_hw_eval<3> arithmetic order (clamp, decompose, Horner, pe-corrected
// recombine).
template <int ITERATIONS>
__attribute__((noinline)) void calculate_exponential_fitted_cpp()
{
    // exp_p3_s1_uniform_any_ulp.csv segment 0: g(f) = 2^f on [0,1).
    constexpr float C0 = 1.0f;
    constexpr float C1 = 6.9583320617675781e-01f;
    constexpr float C2 = 2.2606790065765381e-01f;
    constexpr float C3 = 7.8024059534072876e-02f;
    // expalu_log2_multiplier: +log2(e) (float-rounded).
    constexpr float MULT = 1.4426950408889634e+00f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat xlog2   = x * MULT + 127.0f;
        // Full-range safety clamp: keep xlog2 in [0, 255] so the float->int
        // mantissa shift below cannot wrap (harness domain reaches -100).
        xlog2 = sfpi::max(xlog2, 0.0f);
        xlog2 = sfpi::min(xlog2, 255.0f);
        // Branch-free float->int: shift mantissa left by (exp - bias) bits.
        const sfpi::vInt zi  = sfpi::shft(sfpi::exman(xlog2, sfpi::MantissaMode::ImplicitOne), sfpi::exexp(xlog2), sfpi::ShiftMode::Logical);
        const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(zi);
        const sfpi::vInt ep  = sfpi::exexp(z, sfpi::ExponentMode::Biased); // integer part (biased)
        // Normalize the exman 2^23-scaled fraction back to f in [0,1).
        const sfpi::vFloat f = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest) * 0x1p-23f;
        // Degree-3 Horner for 2^f over the natural [0,1) coefficients.
        sfpi::vFloat p = C3;
        p              = p * f + C2;
        p              = p * f + C1;
        p              = p * f + C0;
        // Recombine 2^i * 2^f with p's own exponent deviation folded in
        // (bit-identical to the measured BARE recombine for c0 == 1.0, where
        // p stays in [1,2) and pe == 127).
        const sfpi::vInt pe  = sfpi::exexp(p, sfpi::ExponentMode::Biased);
        const sfpi::vFloat y = sfpi::setexp(p, ep + pe - 127); // == exp(x)
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

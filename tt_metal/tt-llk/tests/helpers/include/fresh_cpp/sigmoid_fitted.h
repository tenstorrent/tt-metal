// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CM, 2026-08-19).
// Fitted sigmoid vendored from the tt-polynomial-fitter frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/sigmoid_p2_s1_uniform_any_ulp.csv (BH, bf16;
//                  expalu_kind=exp2, expalu_log2_multiplier=-log2(e),
//                  expalu_compose=sigmoid, reduced domain f in [0,1)).
//   kernel shape : tt-metal branch nkapre/tt-polynomial-fitter @ 8063ae8eced6529bd5fa9d8336066601eaa4fd67
//                  tt_metal/programming_examples/generic_lut_activation_embedded/
//                  kernels/compute/piecewise_generic.cpp — exp_hw_eval<2> with
//                  EXPONENT_ALU_EXP2 + EXP_HW_COMPOSE_SIGMOID (non-fused,
//                  pe-corrected setexp recombine, USE_BF16 1-iter reciprocal).
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-19).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners P2/s1):
//   max_ulp_pure_bf16 0.7880, 1.80 us vs TTNN 0.7880 ulp @ 2.81 us.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

// LANE-GC WINNER-REFRESH AUDIT (2026-08-23): still the frontier winner at
// tt-polynomial-fitter origin/main 4cbc636d7fc7202d01a8bdb6ea08eb214445e05e —
// data/coefficients/sigmoid_p2_s1_uniform_any_ulp.csv byte-identical since
// the vendoring sha; tier_silicon_summary.csv sigmoid row = P2/s1, certified
// max_ulp_pure_bf16 0.7880 @ 1.80us vs TTNN 0.7880 @ 2.81us.  Corpus-domain
// re-verification (U[-8, 8] exhaustive bf16, fp64 sigmoid golden, ttpoly
// units.py pure-ULP semantics, two-rounding AND fma_model_bh mad models):
// max 0.7880 — golden agreement exact.  No eltwise-bodies PR filed upstream
// as of 2026-08-23 (lane GB recon).  Coefficients unchanged by lane GC.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted sigmoid (frontier winner sigmoid_p2_s1): sigmoid(x) = 1/(1 + exp(-x))
// with exp(-x) = 2^(x * -log2 e) evaluated by the exponent-ALU decompose —
// integer part into the exponent field, degree-2 natural-basis refinement
// polynomial for 2^f on f in [0,1) — then the compose reciprocal.  Mirrors
// exp_hw_eval<2> arithmetic order exactly (clamp, decompose, Horner,
// pe-corrected recombine); the production sfpu_reciprocal_iter<1> becomes the
// corpus-blessed fresh_recip<1> (same Newton math, all-local constants).
template <int ITERATIONS>
__attribute__((noinline)) void calculate_sigmoid_fitted_cpp()
{
    // sigmoid_p2_s1_uniform_any_ulp.csv segment 0: g(f) = 2^f on [0,1).
    constexpr float C0 = 1.0017247200012207e+00f;
    constexpr float C1 = 6.5763652324676514e-01f;
    constexpr float C2 = 3.3718919754028320e-01f;
    // expalu_log2_multiplier: -log2(e) (float-rounded).
    constexpr float MULT = -1.4426950408889634e+00f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat xlog2   = x * MULT + 127.0f;
        // Full-range safety clamp: keep xlog2 in [0, 255] so the float->int
        // mantissa shift below cannot wrap.
        xlog2 = sfpi::max(xlog2, 0.0f);
        xlog2 = sfpi::min(xlog2, 255.0f);
        // Branch-free float->int: shift mantissa left by (exp - bias) bits.
        const sfpi::vInt zi  = sfpi::shft(sfpi::exman(xlog2, sfpi::MantissaMode::ImplicitOne), sfpi::exexp(xlog2), sfpi::ShiftMode::Logical);
        const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(zi);
        const sfpi::vInt ep  = sfpi::exexp(z, sfpi::ExponentMode::Biased); // integer part (biased)
        // Normalize the exman 2^23-scaled fraction back to f in [0,1).
        const sfpi::vFloat f = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest) * 0x1p-23f;
        // Degree-2 Horner for 2^f over the natural [0,1) coefficients.
        sfpi::vFloat p = C2;
        p              = p * f + C1;
        p              = p * f + C0;
        // Recombine 2^i * 2^f with p's own exponent deviation folded in
        // (g(0)=c0 may sit just off 1.0, so a bare setexp would mis-scale).
        const sfpi::vInt pe  = sfpi::exexp(p, sfpi::ExponentMode::Biased);
        const sfpi::vFloat y = sfpi::setexp(p, ep + pe - 127); // == exp(-x)
        const sfpi::vFloat s = fresh_recip<1>(1.0f + y);
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(s, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

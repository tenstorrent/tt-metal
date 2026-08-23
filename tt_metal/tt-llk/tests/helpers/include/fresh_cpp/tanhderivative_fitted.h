// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CM, 2026-08-19).
// Fitted tanh-derivative vendored from the tt-polynomial-fitter frontier
// selection for tanh_bw (d/dx tanh = 1 - tanh(x)^2 = sech(x)^2):
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/tanh_bw_p4_s8_uniform_any_ulp.csv (BH, bf16)
//   kernel shape : tt-metal branch nkapre/tt-polynomial-fitter @ 8063ae8eced6529bd5fa9d8336066601eaa4fd67
//                  tt_metal/programming_examples/generic_lut_activation_embedded/
//                  kernels/compute/piecewise_generic.cpp — piecewise_generic_lut
//                  poly cascade (segment-0 Horner, then per-segment
//                  v_if(x >= boundary) overwrite; full-degree Horner every
//                  segment, no range reduction, no clamps).
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-19).
//   Recorded claim (silicon BH/BF16 frontier, bf16_bw partition, P4/s8
//   uniform): max_ulp_pure_bf16 0.9114, 7.37 us vs TTNN 1.1615 ulp @ 7.99 us.
//   CONTRACT NOTE: this body targets the TRUE-derivative golden (torch,
//   mathop TanhDerivative).  It must NOT be wired against the
//   tanhderivative-lut row, whose golden IS the production 3-region
//   piecewise-linear LUT — a more accurate kernel fails that golden.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16_bw/summary_bf16.csv selection.

// LANE-GC WINNER-REFRESH AUDIT (2026-08-23): still the frontier winner at
// tt-polynomial-fitter origin/main 4cbc636d7fc7202d01a8bdb6ea08eb214445e05e —
// data/coefficients/tanh_bw_p4_s8_uniform_any_ulp.csv byte-identical since
// the vendoring sha; bf16_bw board pareto_winners tanh_bw row = P4/s8,
// certified max_ulp_pure_bf16 0.9114 @ 7.37us vs TTNN 1.1615 @ 7.99us.
// Corpus-domain re-verification (U[-5, 5] exhaustive bf16, 1 - tanh(x)^2
// fp64 golden, ttpoly units.py pure-ULP semantics, two-rounding AND
// fma_model_bh mad models): max 0.9114 — golden agreement exact.  No
// eltwise-bodies PR filed upstream as of 2026-08-23 (lane GB recon).
// Coefficients unchanged by lane GC.

#include <cstdint>

namespace ckernel::sfpu
{

// Fitted tanh-derivative (frontier winner tanh_bw_p4_s8_uniform): 8-segment
// degree-4 polynomial cascade over [-5, 5] (uniform interior boundaries
// -3.75, -2.5, -1.25, 0, 1.25, 2.5, 3.75).  The fit is even-symmetric
// (mirrored coefficient rows); the measured kernel evaluates it as a plain
// cascade and this body mirrors that arithmetic exactly.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_tanh_derivative_fitted_cpp()
{
    // tanh_bw_p4_s8_uniform_any_ulp.csv rows 0..7: {c0..c4} per segment.
    constexpr float S0[5] = {2.3667706365258970e-01f, 1.8335093334367938e-01f, 5.3897899370391883e-02f, 7.1105891016061277e-03f, 3.5461660189632329e-04f};
    constexpr float S1[5] = {9.5003241036806674e-01f, 9.5899336258861889e-01f, 3.7176749943084636e-01f, 6.5279768584087197e-02f, 4.3635637506903345e-03f};
    constexpr float S2[5] = {1.7028833627700806e+00f, 2.0558052062988281e+00f, 9.6741127967834473e-01f, 2.0766183733940125e-01f, 1.6936123371124268e-02f};
    constexpr float S3[5] = {9.9862476228820229e-01f, -5.7632673230537189e-02f, -1.3915312578519539e+00f, -9.4076244037031331e-01f, -1.8555898672738655e-01f};
    constexpr float S4[5] = {9.9862476228820229e-01f, 5.7632673230537189e-02f, -1.3915312578519539e+00f, 9.4076244037031331e-01f, -1.8555898672738655e-01f};
    constexpr float S5[5] = {1.7028833627700806e+00f, -2.0558052062988281e+00f, 9.6741127967834473e-01f, -2.0766183733940125e-01f, 1.6936123371124268e-02f};
    constexpr float S6[5] = {9.5003241036806674e-01f, -9.5899336258861889e-01f, 3.7176749943084636e-01f, -6.5279768584087197e-02f, 4.3635637506903345e-03f};
    constexpr float S7[5] = {2.3667706365258970e-01f, -1.8335093334367938e-01f, 5.3897899370391883e-02f, -7.1105891016061277e-03f, 3.5461660189632329e-04f};
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat r       = ((((S0[4] * x + S0[3]) * x + S0[2]) * x + S0[1]) * x + S0[0]);
        v_if (x >= -3.75f)
        {
            r = ((((S1[4] * x + S1[3]) * x + S1[2]) * x + S1[1]) * x + S1[0]);
        }
        v_endif;
        v_if (x >= -2.5f)
        {
            r = ((((S2[4] * x + S2[3]) * x + S2[2]) * x + S2[1]) * x + S2[0]);
        }
        v_endif;
        v_if (x >= -1.25f)
        {
            r = ((((S3[4] * x + S3[3]) * x + S3[2]) * x + S3[1]) * x + S3[0]);
        }
        v_endif;
        v_if (x >= 0.0f)
        {
            r = ((((S4[4] * x + S4[3]) * x + S4[2]) * x + S4[1]) * x + S4[0]);
        }
        v_endif;
        v_if (x >= 1.25f)
        {
            r = ((((S5[4] * x + S5[3]) * x + S5[2]) * x + S5[1]) * x + S5[0]);
        }
        v_endif;
        v_if (x >= 2.5f)
        {
            r = ((((S6[4] * x + S6[3]) * x + S6[2]) * x + S6[1]) * x + S6[0]);
        }
        v_endif;
        v_if (x >= 3.75f)
        {
            r = ((((S7[4] * x + S7[3]) * x + S7[2]) * x + S7[1]) * x + S7[0]);
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

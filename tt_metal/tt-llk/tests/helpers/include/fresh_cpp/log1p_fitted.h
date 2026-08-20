// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// PROVENANCE — PLACEHOLDER-PENDING-UPSTREAM-MERGE (lane CR, 2026-08-20).
// Fitted log1p vendored from the tt-polynomial-fitter frontier selection:
//   coefficients : tenstorrent/tt-polynomial-fitter @ 87794c847bc07022de7164f747a9b5d31e3adc47
//                  data/coefficients/log1p_p2_s1_uniform_any_ulp.csv (BH, bf16;
//                  expalu_kind=log1p, Juffa exact-reconstruction reduction,
//                  reduced domain r in [-0.25, 0.5)).
//   kernel shape : same repo/sha deployment/generic_lut_activation/kernels/compute/
//                  piecewise_generic.cpp — log1p_hw_eval<2> with
//                  EXPONENT_ALU_LOG1P (the tt-metal branch
//                  nkapre/tt-polynomial-fitter @ 8063ae8eced6 predates the Juffa
//                  path; the fitter deployment tree carries the measured kernel).
//   NOT YET on tt-metal main (no upstream PR as of 2026-08-20).
//   Recorded claim (silicon BH/BF16 frontier, pareto_winners P2/s1):
//   max_ulp_pure_bf16 0.5942, 1.90 us vs TTNN 0.9310 ulp @ 2.56 us.
//   Same Juffa mechanism as the production kernel and the corpus fresh body
//   (fresh_cpp/log1p.h) — what the fitter contributes is a REFIT of the three
//   correction coefficients P(r) = (log1p(r)-r)/r^2 (production/fresh carry
//   -0x1.008p-1 / 0x1.744p-2 / -0x1p-2).  The IEEE specials guards are
//   SKIPPED as in the measured range-controlled config: the corpus log1p
//   contract is [-0.99, 10] with the golden's (-inf, -1+1e-6) hole excluded,
//   so the decompose input 1+x is always a positive normal.
//   RE-SYNC: when the generic_lut_activation kernels merge upstream or the
//   fitter refits, re-derive from the then-current
//   paper/results/frontier_pareto/silicon/bh/bf16/summary_bf16.csv selection.

#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fitted log1p (frontier winner log1p_p2_s1): Juffa exact-reconstruction —
// u = 1+x is formed ONLY to read the exponent k (isolated on the [0.75, 1.5)
// window); the reduced argument r = 2^-k*x + (2^-k - 1) is rebuilt from the
// ORIGINAL x so no low bits are lost, then
//     log1p(x) = k*ln2 + [ r + r^2 * P(r) ],  P = the fitted correction.
// Mirrors log1p_hw_eval<2> arithmetic order.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_log1p_fitted_cpp()
{
    // log1p_p2_s1_uniform_any_ulp.csv segment 0: P(r) on [-0.25, 0.5).
    constexpr float C0 = -5.0170612335205078e-01f;
    constexpr float C1 = 3.4454023838043213e-01f;
    constexpr float C2 = -2.0109501481056213e-01f;
    // ln(2) * 2^-23: k arrives in exponent-bit units (k << 23).
    constexpr float LN2_EXPBIT = 0.69314718055994530942f * 0x1p-23f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat u = x + 1.0f; // formed only for its exponent

        // e = k << 23: subtract the encoding of 0.75, zero the mantissa bits.
        const sfpi::vFloat three_quarters = 0.75f;
        sfpi::vInt e                      = sfpi::as<sfpi::vInt>(u) - sfpi::as<sfpi::vInt>(three_quarters);
        e                                 = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(e), 0));

        // Reconstruct the reduced argument from the ORIGINAL x:
        //   2^-k * x = reinterpret(bits(x) - e); t = 2^-k - 1 = -0.25*(-4*2^-k) - 1.
        const sfpi::vFloat neg_four    = -4.0f;
        const sfpi::vFloat two_neg_k_x = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(x) - e);
        const sfpi::vFloat s           = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(neg_four) - e); // -4 * 2^-k
        const sfpi::vFloat t           = -0.25f * s - 1.0f;                                          // 2^-k - 1 (exact)
        const sfpi::vFloat r           = two_neg_k_x + t;                                            // 2^-k*(1+x) - 1 in [-0.25, 0.5)

        // log1p(r) = r + r^2 * P(r), fitted natural-basis Horner.
        sfpi::vFloat p      = C2;
        p                   = p * r + C1;
        p                   = p * r + C0;
        sfpi::vFloat result = r + (r * r) * p;

        // + k*ln2: |e| = |k| << 23 converts exactly; restore the sign.
        sfpi::vFloat e_float = sfpi::convert<sfpi::vFloat>(sfpi::abs(e), sfpi::RoundMode::Nearest);
        e_float              = sfpi::copysgn(e_float, sfpi::as<sfpi::vFloat>(e));
        result               = e_float * LN2_EXPBIT + result;

        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

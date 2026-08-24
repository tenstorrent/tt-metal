// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel::sfpu
{

// LICENSED semantic body for the geluappx-fresh row (owner ratification
// 2026-08-24, review_records/OWNER-RATIFICATION-arm-preference-lut-license.md
// item 2: equal-or-better error than the hand kernel on the row's golden
// domain, never worse).
//
// Hand arm = calculate_gelu_appx: SFPLUTFP32 FP16 6-entry TABLE1 (lut2_sign
// over LReg0/1/2/4/5/6, knees 0.5/1/1.5/2/3) computing the even part
// g(|x|) = gelu(x) - 0.5x, plus 0.5x.  Measured max |err| vs exact gelu on
// the row's golden domain (all finite bf16 — the GeluAppx stimulus is an
// UNTRUNCATED Gaussian): 0.023411 raw / 0.023854 bf16-stored, at x ~ 0.249
// (the [0,0.5) segment is a chord through the origin; laneGI accuracy
// oracle, exact fma_model_bh pipeline).
//
// The compiler's LUT selection surface is 3-entry-only (breakpoints exactly
// 1.0/2.0; rvtt-lut-tables.cc) and a 3-piece AFFINE tree cannot reach this
// bar (minimax floor on [0,1) alone is 0.0375 > 0.0234 — laneGI
// fit_licensed.py), so the licensed arm is an independently fitted
// magnitude-dispatch tree with POLYNOMIAL leaves at 8.6x tighter accuracy:
//   [0,1):    degree-3 minimax of g, sup err 2.64e-4
//   [1,2):    degree-2 minimax,      sup err 1.84e-3
//   [2,4):    degree-2 minimax,      sup err 2.72e-3
//   [4,inf):  0.5*a exactly (the asymptote, and the hand kernel's own
//             [3,inf) segment): sup err m*(1-Phi(m)) at m=4 = 1.27e-4,
//             decaying to 0 — pointwise-dominant over any offset form in
//             the bf16-truncation-limited large-|x| regime
// then out = g + 0.5x.  Proven equal-or-better than hand exhaustively over
// the bf16 grid (all mul/add fusion orderings) AND over all fp32 stimuli
// |x| <= 16 under the bf16-RTNE unpack model:
// laneGI-evidence-20260824/accuracy-oracle/verify_arms.c.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_gelu_appx_licensed_cpp()
{
    // NO unroll pragma: measured-negative on this predicated-tree shape
    // (headline-laneGI2-20260824 geluappx-fresh +654.93 unrolled vs
    // headline-laneGI-20260824 +559.07 rolled).
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat a = sfpi::abs(x);
        sfpi::vFloat g       = a * 0.5f;
        v_if (a < 1.0f)
        {
            g = ((a * -0.101991095f + 0.4528101f) * a + -0.009474259f) * a + 0.00026426624f;
        }
        v_elseif (a < 2.0f)
        {
            g = (a * -0.005348735f + 0.63288766f) * a + -0.2880375f;
        }
        v_elseif (a < 4.0f)
        {
            g = (a * -0.019493172f + 0.63692504f) * a + -0.23865677f;
        }
        v_endif;
        sfpi::dst_reg[0] = g + x * 0.5f;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

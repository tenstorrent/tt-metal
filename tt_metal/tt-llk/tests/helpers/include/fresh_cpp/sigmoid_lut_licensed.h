// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel::sfpu
{

// LICENSED semantic body for the sigmoidlut-fresh row (owner ratification
// 2026-08-24, review_records/OWNER-RATIFICATION-arm-preference-lut-license.md
// item 2: equal-or-better error than the hand kernel on the row's golden
// domain, never worse).
//
// Hand arm = the legacy tt-llk _calculate_sigmoid_: SFPLUTFP32 FP16 6-entry
// TABLE2 (mod0 7, knees 0.5/1/1.5/2/4, sign-retain) + 0.5f.  Measured max
// |err| vs exact sigmoid on the row's golden domain (all bf16 in [-8, 8],
// uniform stimuli): 0.017742 raw / 0.017986 bf16-stored, at the |x| = 4
// knee (laneGI accuracy oracle, exact fma_model_bh pipeline).
//
// The compiler's LUT selection surface is 3-entry-only (breakpoints exactly
// 1.0/2.0; rvtt-lut-tables.cc) and a 3-piece AFFINE tree cannot reach this
// bar (minimax floor ~0.028 on [0,1) alone — laneGI fit_licensed.py), so
// the licensed arm is an independently fitted magnitude-dispatch tree with
// POLYNOMIAL leaves at 4.8x tighter accuracy, on the odd part
// s(|x|) = sigmoid(|x|) - 0.5:
//   [0,1):  degree-3 minimax, sup err 3.29e-5
//   [1,2):  degree-2 minimax, sup err 7.98e-5
//   [2,4):  degree-2 minimax, sup err 1.37e-3
//   [4,8]:  affine minimax,   sup err 3.67e-3
// then out = setsgn(s, x) + 0.5.  Proven equal-or-better than hand
// exhaustively over the bf16 grid (all mul/add fusion orderings) AND over
// all fp32 stimuli in [-8, 8] under the bf16-RTNE unpack model:
// laneGI-evidence-20260824/accuracy-oracle/verify_arms.c.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_sigmoid_lut_licensed_cpp()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat a = sfpi::abs(x);
        sfpi::vFloat s       = a * 0.0044123163f + 0.46803683f;
        v_if (a < 1.0f)
        {
            s = ((a * -0.015643528f + -0.004257071f) * a + 0.2509592f) * a + -3.2904663e-05f;
        }
        v_elseif (a < 2.0f)
        {
            s = (a * -0.046415873f + 0.2888266f) * a + -0.011272407f;
        }
        v_elseif (a < 4.0f)
        {
            s = (a * -0.021343574f + 0.1773025f) * a + 0.11293371f;
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::setsgn(s, x) + 0.5f;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

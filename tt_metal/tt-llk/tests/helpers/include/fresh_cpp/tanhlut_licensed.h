// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel::sfpu
{

// LICENSED semantic body for the tanhlut-fresh row (owner ratification
// 2026-08-24, review_records/OWNER-RATIFICATION-arm-preference-lut-license.md
// item 2: approx-contract semantic arms are licensed to MATCH THE HAND ARM'S
// measured accuracy — equal-or-better error on the row's golden domain,
// never worse).
//
// Hand arm = calculate_tanh<APPROX=true>: raw SFPLUT (imm16 tables
// 0x1DFF/0x481A/0xFF00), the 3-region PWL with breakpoints 1.0/2.0 and
// measured max |err| vs exact tanh of 0.144656 at |x| = 1.0 on the row's
// golden domain (all bf16 in [-5, 5]; laneGI accuracy oracle, exact
// fma_model_bh pipeline).
//
// This licensed arm states an INDEPENDENT minimax 3-piece PWL on the same
// breakpoints (the lut-select-formable shape: affine / affine / constant
// magnitude-dispatch tree + setsgn, the tanhderivlut-fresh precedent).
// Fitted max |err| vs exact tanh: 0.041048 (region [0,1) equioscillation) —
// 3.5x TIGHTER than the hand arm, proven exhaustively over the bf16 grid
// under the exact BH SFPU arithmetic model for BOTH emissions (formed
// SFPLUTFP32 mod0=4 and the predicated-tree fallback, every mul/add fusion
// ordering): laneGI-evidence-20260824/accuracy-oracle/verify_arms.c.
//
// Under -mtt-tensix-optimize-lut-select-leaf-ext + -ffinite-math-only (the
// owner-signed licensed leg, tanhderivlut precedent) the constant tail leaf
// admits and the tree forms one SFPLUTFP32 (fp32-3entry-sgn-retain).
template <int ITERATIONS>
__attribute__((noinline)) void calculate_tanh_lut_licensed_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat a = sfpi::abs(x);
        sfpi::vFloat t       = 0.9819684f; // tanh saturation plateau, [2, inf)
        v_if (a < 1.0f)
        {
            t = a * 0.7616004f + 0.041041862f;
        }
        v_elseif (a < 2.0f)
        {
            t = a * 0.20242915f + 0.58077633f;
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::setsgn(t, x);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

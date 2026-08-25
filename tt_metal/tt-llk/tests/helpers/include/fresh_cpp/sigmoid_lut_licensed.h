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
// TABLE2 (mod0 7, knees 0.5/1/1.5/2/4, sign-retain) on the odd part
// s(|x|) = sigmoid(|x|) - 0.5, then + 0.5.  Measured max |err| vs exact
// sigmoid on the row's golden domain: 0.017742 raw / 0.017986 bf16-stored
// on the bf16 grid |x| <= 8, and 0.017742 over ALL fp32 stimuli in [-8, 8]
// (the row's corr format is Float32; laneGI/GU accuracy oracle, exact
// fma_model_bh pipeline).
//
// LANE GU (2026-08-25): the licensed arm is re-expressed in EXACTLY the
// hand kernel's own table geometry — a six-range affine magnitude
// dispatch tree over the architectural TABLE2 breakpoints (0.5/1/1.5/2/4)
// with a trailing setsgn, every coefficient on the SFPLUTFP32 LUT16
// lattice — the shape the FP16 six-entry LUT selection
// (-mtt-tensix-optimize-lut-select-fp16, sfpi-gcc agent/fp16-6entry-lut)
// forms into ONE SFPLUTFP32 mod0 7, the hand kernel's exact instruction.
// Multipartite refit (fixed architectural breakpoints; per-segment
// minimax affine on the LUT16 lattice through the exact fma_model_bh
// pipeline): max |err| vs exact sigmoid 0.010682 raw / 0.012208
// bf16-stored on the bf16 grid, and 0.011150 exhaustively over ALL fp32
// stimuli in [-8, 8] — BEATS the hand bar on every metric
// (laneGU-evidence-20260825/fits/fit_gu6.out).  The tree<->LUT delivery
// is BIT-EXACT on BH for every 2^32 input (all-affine slots, exact LUT16
// re-encode, six-way bucket agreement certified all-2^32:
// laneGU-evidence-20260825/admission-proofs/certifier6-run2.log), so the
// formation needs no finite-math license and the knob leg pairs CRAQ
// bit-exactly with the plain leg.
//
// (The previous licensed arm — a 4-region poly-leaf tree, laneGI — was
// accuracy-passing but MEASURED WORSE than the exact body (+570.60 vs
// +289.78): predicated poly trees lose without LUT formation.  This
// six-range affine form exists to BE formed.)
template <int ITERATIONS>
__attribute__((noinline)) void calculate_sigmoid_lut_licensed_cpp()
{
    // NO unroll pragma (the laneGI geluappx measurement: unroll-8 is
    // measured-negative on predicated-tree shapes).
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat a = sfpi::abs(x);
        // [4, inf)
        sfpi::vFloat s = a * 0x1.214p-8f + 0x1.df4p-2f;
        v_if (a < 0.5f)
        {
            s = a * 0x1.f88p-3f + 0x1p-15f;
        }
        v_elseif (a < 1.0f)
        {
            s = a * 0x1.bd4p-3f + 0x1.e94p-7f;
        }
        v_elseif (a < 1.5f)
        {
            s = a * 0x1.634p-3f + 0x1.e38p-5f;
        }
        v_elseif (a < 2.0f)
        {
            s = a * 0x1.03cp-3f + 0x1.078p-3f;
        }
        v_elseif (a < 4.0f)
        {
            s = a * 0x1.a08p-5f + 0x1.28cp-2f;
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::setsgn(s, x) + 0.5f;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

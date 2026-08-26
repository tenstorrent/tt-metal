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
// LANE GU (2026-08-25; supersedes the laneGI 3-piece PWL, err 0.041048):
// the licensed arm upgrades to a SIX-range affine magnitude dispatch tree
// over the architectural SFPLUTFP32 FP16 six-entry TABLE2 breakpoints
// (0.5/1/1.5/2/4) with a trailing setsgn, every coefficient on the LUT16
// lattice — the shape the FP16 six-entry LUT selection
// (-mtt-tensix-optimize-lut-select-fp16, sfpi-gcc agent/fp16-6entry-lut)
// forms into ONE SFPLUTFP32 mod0 7 (same 4-word loop as the previously
// formed 3-entry mod0 4, at 3.6x tighter accuracy).  Multipartite refit
// (fixed architectural breakpoints; per-segment minimax affine on the
// LUT16 lattice through the exact fma_model_bh pipeline): max |err| vs
// exact tanh 0.011509 raw / 0.013277 bf16-stored on all bf16 in [-5, 5]
// — 12.6x tighter than the hand bar
// (laneGU-evidence-20260825/fits/fit_gu6.out).  The tree<->LUT delivery
// is BIT-EXACT on BH for every 2^32 input (all-affine slots, exact LUT16
// re-encode, six-way bucket agreement certified all-2^32:
// laneGU-evidence-20260825/admission-proofs/certifier6-run2.log), so —
// unlike the 3-piece const-tail form — the formation needs NO
// finite-math license and NO leaf extension, and the knob leg pairs CRAQ
// bit-exactly with the plain leg.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_tanh_lut_licensed_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat a = sfpi::abs(x);
        // [4, inf): tanh saturation with the fitted lattice slope.
        sfpi::vFloat t = a * 0x1.31p-11f + 0x1.fe8p-1f;
        v_if (a < 0.5f)
        {
            t = a * 0x1.e44p-1f + 0x1.2fcp-15f;
        }
        v_elseif (a < 1.0f)
        {
            t = a * 0x1.338p-1f + 0x1.62cp-3f;
        }
        v_elseif (a < 1.5f)
        {
            t = a * 0x1.28p-2f + 0x1.eb4p-2f;
        }
        v_elseif (a < 2.0f)
        {
            t = a * 0x1.e4cp-4f + 0x1.764p-1f;
        }
        v_elseif (a < 4.0f)
        {
            t = a * 0x1.244p-6f + 0x1.dfp-1f;
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::setsgn(t, x);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

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
// LANE GU (2026-08-25; supersedes the laneGI 4-region poly-leaf tree, err
// 0.0097 but +559% delivery): the licensed arm is re-expressed in EXACTLY
// the hand kernel's own table geometry — a six-range affine magnitude
// dispatch tree over the architectural TABLE1 breakpoints (0.5/1/1.5/2/3)
// whose coefficients all lie on the SFPLUTFP32 LUT16 lattice — the shape
// the FP16 six-entry LUT selection (-mtt-tensix-optimize-lut-select-fp16,
// sfpi-gcc agent/fp16-6entry-lut) forms into ONE SFPLUTFP32 mod0 2, the
// hand kernel's exact instruction.  Multipartite refit (fixed
// architectural breakpoints; per-segment minimax affine on the LUT16
// lattice through the exact fma_model_bh pipeline, both fusion
// orderings): max |err| vs exact gelu over all finite bf16 stimuli
// 0.017778 raw / 0.018187 bf16-stored — BEATS the hand bar on both
// metrics (laneGU-evidence-20260825/fits/fit_gu6.out).  The tree<->LUT
// delivery is BIT-EXACT on BH for every 2^32 input (all-affine slots,
// exact LUT16 re-encode, six-way bucket agreement certified all-2^32:
// laneGU-evidence-20260825/admission-proofs/certifier6-run2.log), so the
// formation needs no finite-math license and the knob leg pairs CRAQ
// bit-exactly with the plain leg.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_gelu_appx_licensed_cpp()
{
    // The 0.5 of the final "+ 0.5x" is parked in the programmable
    // constant register — the HAND kernel's own idiom (its init loads
    // vConstFloatPrgm0 = 0.5).  With it in an LREG the row needs a 9th
    // live LREG (6 packed table words + x + the half + the LUT result),
    // so the formed LUT's transactional coefficient hoist refuses on
    // pressure and the packed words reload per row; as a CReg operand
    // the MAD reads it directly and the whole loop fits the 8-LREG file
    // (laneGU measurement: 71223 -> hand-shape loop).
    sfpi::vConstFloatPrgm0 = 0.5f;
    // NO unroll pragma: measured-negative on this predicated-tree shape
    // (headline-laneGI2-20260824 geluappx-fresh +654.93 unrolled vs
    // headline-laneGI-20260824 +559.07 rolled).
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat a = sfpi::abs(x);
        // [3, inf): the asymptote slope with the fitted lattice intercept.
        sfpi::vFloat g = a * 0x1p-1f + -0x1.2fcp-15f;
        v_if (a < 0.5f)
        {
            g = a * 0x1.58p-3f + -0x1.2fcp-15f;
        }
        v_elseif (a < 1.0f)
        {
            g = a * 0x1.f68p-2f + -0x1.404p-3f;
        }
        v_elseif (a < 1.5f)
        {
            g = a * 0x1.3cp-1f + -0x1.1cp-2f;
        }
        v_elseif (a < 2.0f)
        {
            g = a * 0x1.384p-1f + -0x1.0ep-2f;
        }
        v_elseif (a < 3.0f)
        {
            g = a * 0x1.158p-1f + -0x1p-3f;
        }
        v_endif;
        sfpi::dst_reg[0] = g + x * sfpi::vConstFloatPrgm0;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

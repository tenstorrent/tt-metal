// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic bodies for the float comparison-to-zero ops (storm
// contract: fresh_cpp/README.md).  Independent derivation from the
// mathematical definition (the production golden, golden_generators.py:
// torch.ne/lt/gt/le/ge against 0 -> 1.0 where the comparison holds, else
// 0.0).  The production float path (metal ckernel_sfpu_comp.h
// calculate_comp) is an all-raw-TTI handwritten kernel (SFPSETSGN /
// SFPSETCC / SFPIADD-against-inf choreography) — these bodies are the
// semantic arm it never had (the eqz-fresh / laneBR
// calculate_eqz_fresh_cpp precedent, extended to the remaining five float
// comparisons; laneED sem-only audit).
//
// Result materialization (lane GH 2026-08-24 rewrite; the laneCL eqz-fresh
// dual-store precedent — previous register-materialized bodies preserved in
// fresh_cpp/comp_legacy.h, unwired): every mode stores its DEFAULT answer
// straight from a hard constant register (vConst0/vConst1) and then
// overwrites under the deciding CC — no lane register ever materializes the
// result, so the NotEqualZero vehicle costs exactly the production kernel's
// 6 issue slots/row (load, store-default, abs, setcc, store-cc, encc)
// instead of the legacy 7 (the two SFPMOV result selects).  The rewrite is
// value-preserving by construction: per mode, the predicates and the two
// stored constants are the legacy body's own — only the materialization
// path (predicated CREG store vs predicated CREG->LREG move + store)
// changes (LANEGH-PROOFS.md, laneGH-evidence-20260824).
//
// -0.0 discipline (the eqz-fresh / lane CL rule): every branch decides the
// "is zero" question through sfpi::abs(v) == 0.0f rather than a raw sign
// compare, so both zeros land on the golden's answer (torch: -0.0 == 0).
//
// NaN discipline (lane GH finding GH-F1, 2026-08-24): the edges legs inject
// IEEE specials (specials_safe at Float32/dest_acc=Yes), and the sfpi
// `v > 0.0f` lowering is a sign-clear+bits-nonzero SFPSETCC pair that
// answers TRUE for quiet NaN — the legacy gtz/gez bodies therefore answered
// 1.0 where the golden (torch: any comparison with NaN is False) and the
// production kernel answer 0.0/gez-0.0.  Latent, pre-existing: those edge
// nodes belong to no sweep row and had never been executed (sim-verified at
// canon 9c9d15645c, laneGH-evidence-20260824).  gtz/gez now overwrite the
// answer to 0 under sfpi::is_nan(v) as the last store; ltz/lez/ne are
// naturally NaN-correct under their lowerings (sign test excludes
// sign-clear qNaN; ne's golden answer for NaN is 1, which is its default
// store).  NaN remains outside the functional legs' swept domain
// (uniform(-2, 2) stimuli, sfpu_domains.py).
//
// Rows are addressed by immediate offset (dst_reg[d], full unroll) rather
// than dst_reg++: constant dst indices need no TTINCRWC counter words (the
// calculate_eqz_fresh_cpp convention).

#include <cstdint>

namespace ckernel::sfpu
{

template <SfpuType COMP_MODE, int ITERATIONS>
__attribute__((noinline)) void calculate_comp_fresh_cpp()
{
    static_assert(
        COMP_MODE == SfpuType::not_equal_zero || COMP_MODE == SfpuType::less_than_zero || COMP_MODE == SfpuType::greater_than_zero ||
            COMP_MODE == SfpuType::less_than_equal_zero || COMP_MODE == SfpuType::greater_than_equal_zero,
        "float zero-comparison semantic body; equal_zero is calculate_eqz_fresh_cpp (laneBR)");
#pragma GCC unroll 32
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[d];
        if constexpr (COMP_MODE == SfpuType::not_equal_zero)
        {
            // ne(v, 0) = !(|v| == 0): default 1, both zeros overwrite to 0.
            sfpi::dst_reg[d] = sfpi::vConst1;
            v_if (sfpi::abs(v) == 0.0f)
            {
                sfpi::dst_reg[d] = sfpi::vConst0;
            }
            v_endif;
        }
        else if constexpr (COMP_MODE == SfpuType::less_than_zero)
        {
            // lt(v, 0): true negatives only — a sign-magnitude -0.0 must not
            // count, so the zero test overwrites last.
            sfpi::dst_reg[d] = sfpi::vConst0;
            v_if (v < 0.0f)
            {
                sfpi::dst_reg[d] = sfpi::vConst1;
            }
            v_endif;
            v_if (sfpi::abs(v) == 0.0f)
            {
                sfpi::dst_reg[d] = sfpi::vConst0;
            }
            v_endif;
        }
        else if constexpr (COMP_MODE == SfpuType::greater_than_zero)
        {
            sfpi::dst_reg[d] = sfpi::vConst0;
            v_if (v > 0.0f)
            {
                sfpi::dst_reg[d] = sfpi::vConst1;
            }
            v_endif;
            v_if (sfpi::abs(v) == 0.0f)
            {
                sfpi::dst_reg[d] = sfpi::vConst0;
            }
            v_endif;
            // GH-F1: the > lowering admits quiet NaN; golden answers 0.
            v_if (sfpi::is_nan(v))
            {
                sfpi::dst_reg[d] = sfpi::vConst0;
            }
            v_endif;
        }
        else if constexpr (COMP_MODE == SfpuType::less_than_equal_zero)
        {
            // le(v, 0) = lt(v, 0) || (|v| == 0); both zeros answer 1.
            sfpi::dst_reg[d] = sfpi::vConst0;
            v_if (v < 0.0f)
            {
                sfpi::dst_reg[d] = sfpi::vConst1;
            }
            v_endif;
            v_if (sfpi::abs(v) == 0.0f)
            {
                sfpi::dst_reg[d] = sfpi::vConst1;
            }
            v_endif;
        }
        else // greater_than_equal_zero
        {
            // ge(v, 0) = !(v < 0) with both zeros answering 1 regardless of
            // their sign bit.
            sfpi::dst_reg[d] = sfpi::vConst1;
            v_if (v < 0.0f)
            {
                sfpi::dst_reg[d] = sfpi::vConst0;
            }
            v_endif;
            v_if (sfpi::abs(v) == 0.0f)
            {
                sfpi::dst_reg[d] = sfpi::vConst1;
            }
            v_endif;
            // GH-F1: the default-1 shape admits quiet NaN; golden answers 0.
            v_if (sfpi::is_nan(v))
            {
                sfpi::dst_reg[d] = sfpi::vConst0;
            }
            v_endif;
        }
    }
}

} // namespace ckernel::sfpu

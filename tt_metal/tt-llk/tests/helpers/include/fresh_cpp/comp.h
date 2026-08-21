// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic bodies for the float comparison-to-zero ops (storm
// contract: fresh_cpp/README.md).  Independent derivation from the
// mathematical definition (the production golden, golden_generators.py:
// torch.ne/lt/gt/le/ge against 0 -> 1.0 where the comparison holds, else
// 0.0).  The production float path (metal ckernel_sfpu_comp.h
// calculate_comp) is an all-raw-TTI handwritten kernel (SFPSETSGN /
// SFPSETCC / SFPIADD-against-inf choreography) — the laneED sem-only audit
// found the comp corpus row racing that hand kernel under the "semantic"
// label; these bodies are the semantic arm it never had (the eqz-fresh /
// laneBR calculate_eqz_fresh_cpp precedent, extended to the remaining five
// float comparisons).
//
// -0.0 discipline (the eqz-fresh / lane CL rule): every branch decides the
// "is zero" question through sfpi::abs(v) == 0.0f rather than a raw sign
// compare, so both zeros land on the golden's answer (torch: -0.0 == 0).
// NaN is outside the swept domain (uniform(-2, 2) stimuli, sfpu_domains.py);
// the unarycomp.h precedent note applies unchanged.

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
    // Full unroll + immediate row addressing, the calculate_eqz_fresh_cpp
    // convention: constant dst_reg[d] indices need no TTINCRWC counter words.
#pragma GCC unroll 32
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[d];
        sfpi::vFloat r       = sfpi::vConst0;
        if constexpr (COMP_MODE == SfpuType::not_equal_zero)
        {
            // ne(v, 0) = !(|v| == 0)
            r = sfpi::vConst1;
            v_if (sfpi::abs(v) == 0.0f)
            {
                r = sfpi::vConst0;
            }
            v_endif;
        }
        else if constexpr (COMP_MODE == SfpuType::less_than_zero)
        {
            // lt(v, 0): true negatives only — a sign-magnitude -0.0 must not count.
            v_if (v < 0.0f)
            {
                r = sfpi::vConst1;
            }
            v_endif;
            v_if (sfpi::abs(v) == 0.0f)
            {
                r = sfpi::vConst0;
            }
            v_endif;
        }
        else if constexpr (COMP_MODE == SfpuType::greater_than_zero)
        {
            v_if (v > 0.0f)
            {
                r = sfpi::vConst1;
            }
            v_endif;
            v_if (sfpi::abs(v) == 0.0f)
            {
                r = sfpi::vConst0;
            }
            v_endif;
        }
        else if constexpr (COMP_MODE == SfpuType::less_than_equal_zero)
        {
            // le(v, 0) = lt(v, 0) || (|v| == 0); both zeros answer 1.
            v_if (v < 0.0f)
            {
                r = sfpi::vConst1;
            }
            v_endif;
            v_if (sfpi::abs(v) == 0.0f)
            {
                r = sfpi::vConst1;
            }
            v_endif;
        }
        else // greater_than_equal_zero
        {
            // ge(v, 0) = !(v < 0) with both zeros answering 1 regardless of
            // their sign bit.
            r = sfpi::vConst1;
            v_if (v < 0.0f)
            {
                r = sfpi::vConst0;
            }
            v_endif;
            v_if (sfpi::abs(v) == 0.0f)
            {
                r = sfpi::vConst1;
            }
            v_endif;
        }
        sfpi::dst_reg[d] = r;
    }
}

} // namespace ckernel::sfpu

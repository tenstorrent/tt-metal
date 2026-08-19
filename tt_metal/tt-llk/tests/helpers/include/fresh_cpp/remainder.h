// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the remainder op (storm contract, migrated
// verbatim from fresh_cpp_operations.h, Lane BR batch 2).  The fmod /
// remainder shared core (trunc-by-round + residual mop-up + zero snap)
// lives in fresh_cpp/helpers.h; the divisor sign fold below is the
// torch.remainder contract.
#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_remainder_fresh_cpp(const float divisor, const float divisor_recip)
{
    const sfpi::vFloat divisor_v = divisor;
    const sfpi::vFloat s         = sfpi::abs(divisor_v);
    const sfpi::vFloat recip     = sfpi::abs(sfpi::vFloat(divisor_recip));
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat v         = fresh_fmod_core(sfpi::abs(val), s, recip);
        // remainder folds onto the divisor's sign (torch.remainder contract).
        v_if (val < 0.0f && v != 0.0f)
        {
            v = s - v;
        }
        v_endif;
        v_if (divisor_v < 0.0f && v != 0.0f)
        {
            v = v + divisor_v;
        }
        v_endif;
        v = sfpi::copysgn(v, divisor_v);
        v_if (s == 0.0f)
        {
            v = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
        v_if (sfpi::abs(v) - s == 0.0f)
        {
            v = 0.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

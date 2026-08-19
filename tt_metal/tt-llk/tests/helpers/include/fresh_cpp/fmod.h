// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// fmod — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// Migrated verbatim from ../fresh_cpp_operations.h (Lane BR causal-tier lift);
// depends on fresh_trunc_magnitude/fresh_fmod_core, which stay in
// fresh_cpp_operations.h (shared with the legacy remainder body).
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_fmod_fresh_cpp(const float divisor, const float divisor_recip)
{
    const sfpi::vFloat s     = sfpi::abs(sfpi::vFloat(divisor));
    const sfpi::vFloat recip = sfpi::abs(sfpi::vFloat(divisor_recip));
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat v         = fresh_fmod_core(sfpi::abs(val), s, recip);
        // fmod keeps the dividend's sign.
        v = sfpi::copysgn(v, val);
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

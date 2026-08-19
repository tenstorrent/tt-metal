// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// hardmish — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// hardmish(x) = x * clamp(0.5*x + 1, 0, 1) (the published hard-mish formula;
// golden_generators._hardmish states exactly this).  Production: metal
// calculate_hardmish.  Fresh statement: one affine term and a typed min/max
// clamp.
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_hardmish_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat t       = v * 0.5f + 1.0f;
        t                    = sfpi::min(t, 1.0f);
        t                    = sfpi::max(t, 0.0f);
        sfpi::dst_reg[0]     = v * t;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `add1` corpus row (metal
// calculate_add1).  Mathematical definition: add1(x) = x + 1 (the golden's
// exact statement), one fp32 add per lane.
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_add1_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0]     = v + 1.0f;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `abs` corpus row (legacy
// _calculate_abs_ float path).  Mathematical definition (torch.abs):
// |x|, sign bit cleared, magnitude untouched — exact for every input,
// so no store-rounding statement is needed.
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_abs_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0]     = sfpi::abs(v);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

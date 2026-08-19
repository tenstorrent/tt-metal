// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `bitwisenot` corpus row (metal
// calculate_bitwise_not).  Mathematical definition (torch.bitwise_not over
// int32): ~x, i.e. the exact two's-complement identity ~x = -x - 1 stated at
// the value level; the typed vInt Dst view carries the representation
// contract.
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_bitwise_not_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vInt v = sfpi::dst_reg[0];
        // ~x == (-1) - x in two's complement.
        sfpi::dst_reg[0] = sfpi::vInt(-1) - v;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_square_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat in     = sfpi::dst_reg[0];
        sfpi::vFloat result = in * in;

        sfpi::dst_reg[0] = result;

        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

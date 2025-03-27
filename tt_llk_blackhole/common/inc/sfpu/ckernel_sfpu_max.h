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
inline void _calculate_max_(const int iterations)
{
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[32];
        v_if (a < b)
        {
            sfpi::dst_reg[0] = b;
        }
        v_endif;

        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

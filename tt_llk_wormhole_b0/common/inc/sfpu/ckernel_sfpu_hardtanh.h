// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_hardtanh_(const int iterations, uint param0, uint param1, uint param2)
{
    // All params are in FP16_B format
    // param0 = -(neg_threshold)
    // param1 = -(pos_threshold - neg_threshold)
    // param2 = -(pos_threshold)

    sfpi::vFloat p0 = sfpi::s2vFloat16(param0);
    sfpi::vFloat p1 = sfpi::s2vFloat16(param1);
    sfpi::vFloat p2 = sfpi::s2vFloat16(param2);
// SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];

        val += p0; // 12 bits
        v_if (val < 0.0f)
        {
            val = 0.0f;
        }
        v_endif;

        val += p1; // 12 bits
        v_if (val >= 0.0f)
        {
            val = 0.0f;
        }
        v_endif;

        val += p2; // 12 bits

        sfpi::dst_reg[0] = val;

        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

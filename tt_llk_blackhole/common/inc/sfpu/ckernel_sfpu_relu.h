// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_converter.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, uint slope)
{
    sfpi::vFloat s = Converter::as_float(slope);

#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];

        v_if (v < 0.0f)
        {
            v *= s;
        }
        v_endif;

        sfpi::dst_reg[0] = v;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_(const int iterations, uint uint_threshold)
{
    sfpi::vFloat threshold = sfpi::s2vFloat16(uint_threshold, sfpi::s2vFloat16::fp16a);
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat a = sfpi::dst_reg[0];
        v_if (a > threshold)
        {
            a = threshold;
        }
        v_endif;
        v_if (a < 0.0f)
        {
            a = 0.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = a;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_(const int iterations, uint uint_threshold)
{
    sfpi::vFloat threshold = sfpi::s2vFloat16(uint_threshold, sfpi::s2vFloat16::fp16a);
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat a = sfpi::dst_reg[0];
        v_if (a < threshold)
        {
            a = 0.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = a;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_lrelu_(const int iterations, uint slope)
{
    // SFPU microcode
    vFloat s = s2vFloat16b(slope);

    #pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) {
        vFloat v = dst_reg[0];

        v_if (v < 0.0f) {
            v *= s;
        }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_(const int iterations, uint uint_threshold)
{
    vFloat threshold = s2vFloat16(uint_threshold, s2vFloat16::fp16a);
    for (int d = 0; d < iterations; d++)
    {
        vFloat a = dst_reg[0];
        v_if(a > threshold) {
            a = threshold;
        }
        v_endif;
        v_if(a < 0.0f) {
            a = 0.0f;
        }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_(const int iterations, uint uint_threshold)
{
    vFloat threshold = s2vFloat16(uint_threshold, s2vFloat16::fp16a);
    for (int d = 0; d < iterations; d++)
    {
        vFloat a = dst_reg[0];
        v_if(a < threshold) {
            a = 0.0f;
        }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

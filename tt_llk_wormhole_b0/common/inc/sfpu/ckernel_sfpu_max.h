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
inline void _calculate_max_(const int iterations)
{
    for (int d = 0; d < iterations; d++)
    {
        vFloat a = dst_reg[0];
        vFloat b = dst_reg[32];
        v_if(a < b) {
            dst_reg[0] = b;
        }
        v_endif;

        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

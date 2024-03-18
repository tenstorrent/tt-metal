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
inline void _calculate_max_int32_(const int iterations)
{
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(2, 12, 3, 0);
        TTI_SFPLOAD(0, 12, 3, 64);
        TTI_SFPMOV(0, 0, 1, 0);
        TTI_SFPIADD(0, 2, 1, 2);
        TTI_SFPSTORE(0, 12, 3, 0);
        TTI_SFPENCC(0x003, 0, 0, 10);
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

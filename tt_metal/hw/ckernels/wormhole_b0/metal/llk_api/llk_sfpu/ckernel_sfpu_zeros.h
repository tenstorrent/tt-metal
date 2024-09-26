// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_zeros()
{
    vFloat fill_val = 0.0f;
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        dst_reg[0] = fill_val;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

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
inline void calculate_add1()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        dst_reg[0] = 1.0f + val;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void add1_init()
{
    ;
}

}  // namespace sfpu
}  // namespace ckernel

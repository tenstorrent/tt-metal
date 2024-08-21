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

// TODO: Implement using bitwise comparision
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_signbit()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        v_if (val < 0.0f) {
            val = 1.0f;
        } v_else {
            val = 0.0f;
        }
        v_endif;
        dst_reg[0] = val;

       dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

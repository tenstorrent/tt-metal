// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"

#include "sfpi.h"

#include "ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS=4>
inline void calculate_rsub(uint value)
{
    Converter c_value;
    c_value.u = value;
    vFloat arg2 = c_value.f;

    #pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat value = dst_reg[0];
        dst_reg[0] = arg2 - value;
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel

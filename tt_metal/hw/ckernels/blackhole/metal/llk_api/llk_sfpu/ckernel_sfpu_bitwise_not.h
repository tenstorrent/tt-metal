// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void bitwise_not_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitwise_not() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = ~v;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

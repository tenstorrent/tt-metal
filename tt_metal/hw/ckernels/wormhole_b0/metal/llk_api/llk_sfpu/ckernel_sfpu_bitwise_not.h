// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitwise_not() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load/store as two's-complement int32 (sfpi DataLayout::I32 == InstrModLoadStore::INT32),
        // matching the original TTI path. A plain vInt load would read the raw sign-magnitude dest
        // bits and ~ would then flip the sign-magnitude representation instead of the integer value.
        vInt v = dst_reg[0].mode<sfpi::DataLayout::I32>();
        dst_reg[0].mode<sfpi::DataLayout::I32>() = ~v;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

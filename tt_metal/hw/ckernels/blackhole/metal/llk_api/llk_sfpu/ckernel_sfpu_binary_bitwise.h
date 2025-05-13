// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <
    bool APPROXIMATION_MODE,
    BinaryBitwiseOp BITWISE_OP,
    InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32,
    int ITERATIONS = 8>
inline void calculate_sfpu_binary_bitwise(const uint dst_offset) {
    _calculate_sfpu_binary_bitwise_<APPROXIMATION_MODE, BITWISE_OP, INSTRUCTION_MODE, ITERATIONS>(dst_offset);
}

}  // namespace sfpu
}  // namespace ckernel

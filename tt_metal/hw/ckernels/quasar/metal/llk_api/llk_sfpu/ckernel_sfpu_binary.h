// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_binary.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATE, BinaryOp BINOP, bool EN_32BIT_DEST = false>
inline void calculate_sfpu_binary_div(
    const int iterations,
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_out) {
    _calculate_sfpu_binary_<APPROXIMATE, BINOP, EN_32BIT_DEST>(iterations, dst_index_in0, dst_index_in1, dst_index_out);
}

template <bool APPROXIMATE, BinaryOp BINOP>
inline void sfpu_binary_init() {
    _sfpu_binary_init_<APPROXIMATE, BINOP>();
}

}  // namespace sfpu
}  // namespace ckernel

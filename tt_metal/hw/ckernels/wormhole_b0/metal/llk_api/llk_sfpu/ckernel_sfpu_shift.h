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

template <bool APPROXIMATION_MODE, bool SHIFT_RIGHT, int ITERATIONS = 8>
inline void calculate_shift(const uint dst_offset) {
    _calculate_shift_<APPROXIMATION_MODE, SHIFT_RIGHT, ITERATIONS>(dst_offset);
}

}  // namespace sfpu
}  // namespace ckernel

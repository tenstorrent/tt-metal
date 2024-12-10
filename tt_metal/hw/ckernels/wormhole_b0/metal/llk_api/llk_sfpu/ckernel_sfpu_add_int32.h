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

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_add_int32(const uint dst_offset) {
    _add_int32_<APPROXIMATION_MODE, ITERATIONS>(dst_offset);
}

}  // namespace sfpu
}  // namespace ckernel

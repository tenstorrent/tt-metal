// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool SIGN_MAGNITUDE_FORMAT, int ITERATIONS = 8>
inline void calculate_sub_int32(const uint dst_offset) {
    _sub_int32_<APPROXIMATION_MODE, SIGN_MAGNITUDE_FORMAT, ITERATIONS>(dst_offset);
}

}  // namespace sfpu
}  // namespace ckernel

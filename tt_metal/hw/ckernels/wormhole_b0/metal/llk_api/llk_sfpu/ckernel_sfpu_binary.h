// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int BINOP_MODE, int ITERATIONS = 8>
inline void calculate_sfpu_binary(const uint dst_offset)
{
    _calculate_sfpu_binary_<APPROXIMATION_MODE, BINOP_MODE, ITERATIONS>(dst_offset);
}

}  // namespace sfpu
}  // namespace ckernel

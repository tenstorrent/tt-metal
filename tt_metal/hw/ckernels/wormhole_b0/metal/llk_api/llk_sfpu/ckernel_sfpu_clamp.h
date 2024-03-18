// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_clamp.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_clamp(uint param0, uint param1, uint param2)
{
    _calculate_clamp_<APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, param0, param1, param2);
}

}  // namespace sfpu
}  // namespace ckernel

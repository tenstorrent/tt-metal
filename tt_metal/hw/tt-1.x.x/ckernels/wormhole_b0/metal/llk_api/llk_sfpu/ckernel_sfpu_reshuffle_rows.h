// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_reshuffle_rows.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_reshuffle_rows(uint idx_addr) {
    _calculate_reshuffle_rows_<APPROXIMATION_MODE, ITERATIONS>(idx_addr);
}

}  // namespace sfpu
}  // namespace ckernel

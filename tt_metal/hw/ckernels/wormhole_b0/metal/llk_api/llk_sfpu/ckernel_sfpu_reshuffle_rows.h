// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_reshuffle_rows.h"
#include "llk_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_reshuffle_rows(uint idx_addr) {
    _calculate_reshuffle_rows_<(APPROX_MODE == ApproximationMode::Fast), ITERATIONS>(idx_addr);
}

}  // namespace sfpu
}  // namespace ckernel

// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_reshuffle_rows.h"
#include "llk_defs.h"

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_reshuffle_rows(uint idx_addr) {
    _calculate_reshuffle_rows_<(APPROX_MODE), ITERATIONS>(idx_addr);
}

}  // namespace sfpu
}  // namespace ckernel

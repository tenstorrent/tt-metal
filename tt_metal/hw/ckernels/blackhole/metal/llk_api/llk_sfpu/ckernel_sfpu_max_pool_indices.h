// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpu/ckernel_sfpu_max_pool_indices.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int num_rows, int ITERATIONS = 8>
inline void calculate_max_pool_with_indices(uint values_tile_idx, uint indices_tile_idx, uint tile_idx) {
    _calculate_max_pool_with_indices_<APPROXIMATION_MODE, is_fp32_dest_acc_en, num_rows, ITERATIONS>(
        values_tile_idx, indices_tile_idx, tile_idx);
}

template <bool APPROXIMATION_MODE>
inline void init_max_pool_with_indices() {
    _init_max_pool_with_indices_();
}

}  // namespace sfpu
}  // namespace ckernel

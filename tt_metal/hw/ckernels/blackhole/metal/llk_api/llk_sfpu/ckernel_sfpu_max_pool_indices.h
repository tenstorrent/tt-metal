// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "sfpu/ckernel_sfpu_max_pool_indices.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <
    ApproximationMode APPROX_MODE,
    bool is_fp32_dest_acc_en,
    int num_rows,
    int ITERATIONS = 8,
    DataLayout layout = DataLayout::TILE,
    bool accumulate = false>
inline void calculate_max_pool_with_indices(uint values_tile_idx, uint indices_tile_idx, uint chunk) {
    if constexpr (num_rows <= 9) {
        _calculate_max_pool_with_indices_<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS, layout, accumulate>(
            values_tile_idx, indices_tile_idx, chunk);
    } else {
        static_assert(num_rows <= 32, "num_rows must be <= 32");
        static_assert(
            layout == DataLayout::ROW_MAJOR, "generic max pool with indices is only implemented for ROW_MAJOR layout");
        _calculate_max_pool_with_indices_generic_<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS, accumulate>(
            values_tile_idx, indices_tile_idx, chunk);
    }
}

template <ApproximationMode APPROX_MODE, DataLayout layout = DataLayout::TILE>
inline void init_max_pool_with_indices() {
    _init_max_pool_with_indices_<layout>();
}

}  // namespace sfpu
}  // namespace ckernel

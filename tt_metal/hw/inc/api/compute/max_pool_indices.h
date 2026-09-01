// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#ifndef ARCH_QUASAR
#ifdef TRISC_MATH
#include "ckernel_sfpu_max_pool_indices.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs MaxPool with indices algorithm on the data tile and index tile
 * that are pre-loaded in DST register. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * | Argument        | Description                                                                 | Type       | Valid Range                                           | Required |
 * |-----------------|-----------------------------------------------------------------------------|------------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register containing the data to be reduced     | uint32_t   | Must be less than the size of the DST register buffer | True     |
 * | idst_idx        | The index of the tile in DST register containing the indices of the data    | uint32_t   | Must be less than the size of the DST register buffer | True     |
 * | chunk           | The index of the intra-kernel "chunk" of data for large kernel accumulation | uint32_t   | 0 to UINT_MAX                                         | False    |
 * | num_rows        | The number of rows to use for the MaxPool operation                         | uint32_t   | <= 32, but note either 9 or 32 rows will be reduced   | False    |
 * | layout          | The data layout of the data in DST                                          | DataLayout | TILE or ROW_MAJOR                                     | False    |
 * | accumulate      | Whether to accumulate results for large kernels                             | bool       | true, false                                           | False    |
 * | ITERATIONS      | The number of iterations to perform (unused)                                | int        | 1 to 8                                                | False    |
 */
// clang-format on
template <
    int num_rows = 9,
    ckernel::DataLayout layout = ckernel::DataLayout::TILE,
    bool accumulate = false,
    int ITERATIONS = 8,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void max_reduce_with_indices(uint32_t idst, uint32_t idst_idx, uint32_t chunk = 0) {
    static_assert(num_rows <= 32, "num_rows must be <= 32");
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_max_pool_with_indices,
        (true /* APPROXIMATE */, is_fp32_dest_acc_en, num_rows, ITERATIONS, layout, accumulate),
        idst,
        idst_idx,
        0 /* DST out unused, but required for _llk_math_eltwise_binary_sfpu_params_ */,
        VectorMode::RC,
        chunk)));
}

/**
 * Please refer to documentation for any_init.
 */
template <ckernel::DataLayout layout = ckernel::DataLayout::TILE>
ALWI void max_reduce_with_indices_init() {
    MATH((SFPU_BINARY_INIT_FN(
        max_pool_with_indices, sfpu::init_max_pool_with_indices, (true /* APPROXIMATE */, layout))));
}

}  // namespace ckernel

#endif  // !ARCH_QUASAR

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#ifndef ARCH_QUASAR
#ifdef TRISC_MATH
#include "ckernel_sfpu_add_top_row.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise add_top_row operation between the top rows of two tiles in DST register.
 * Takes the top row of tile at dst_tile_0 and adds it with the top row of tile at dst_tile_1,
 * storing the result in the top row of tile at dst_tile_out.
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Only 32x32 tile dimensions are supported.
 * All tile indices must reference valid tiles within the DST register.
 *
 * | Argument        | Description                                                              | Type      | Valid Range                                           | Required |
 * |-----------------|--------------------------------------------------------------------------|-----------|-------------------------------------------------------|----------|
 * | dst_tile_0      | The index of the first tile in DST register                              | uint32_t  | Must be less than the size of the DST register buffer | True     |
 * | dst_tile_1      | The index of the second tile in DST register                             | uint32_t  | Must be less than the size of the DST register buffer | True     |
 * | dst_tile_out    | The index of the output tile in DST register                             | uint32_t  | Must be less than the size of the DST register buffer | True     |
 * | format          | The data format for the add_top_row operation                            | DataFormat| Float32, Int32, UInt32                                | True     |
 */
// clang-format on
template <DataFormat format>
ALWI void sfpu_add_top_row(uint32_t dst_tile_0, uint32_t dst_tile_1, uint32_t dst_tile_out) {
    static_assert(
        format == DataFormat::Float32 || format == DataFormat::Int32 || format == DataFormat::UInt32,
        "Unsupported data format. Supported formats: Float32, Int32, UInt32");

    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_add_top_row,
        (format),
        dst_tile_0,
        dst_tile_1,
        dst_tile_out,
        VectorMode::RC_custom)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void sfpu_add_top_row_init() { MATH((SFPU_BINARY_INIT_FN_NO_ARGS(add_top_row, sfpu::init_add_top_row))); }

}  // namespace ckernel

#endif  // !ARCH_QUASAR

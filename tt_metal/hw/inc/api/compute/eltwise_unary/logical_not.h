// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_logical_not.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {
// clang-format off
/**
 * Performs element-wise computation of the logical not operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * @tparam data_format Template argument specifying the data type.
 * Supported data formats are: DataFormat::Int32, DataFormat::UInt32, DataFormat::UInt16, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat DATA_FORMAT>
ALWI void logical_not_tile(uint32_t idst) {
    MATH(SFPU_UNARY_KERNEL_THREE_TEMPLATE_ARGS_FN(calculate_logical_not, APPROX, DATA_FORMAT, 8, idst, RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void logical_not_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(logical_not_unary, APPROX)); }

}  // namespace ckernel

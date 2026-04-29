// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_binary_comp.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise comparison operation with two integer inputs: y = comparison_op(x0,x1)
 * Supports Int32, UInt32 and UInt16 data formats (selected via the data_format template parameter).
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * Return value: None
 *
 * | Template Param | Description                                                           | Valid Values                             | Required |
 * |----------------|-----------------------------------------------------------------------|------------------------------------------|----------|
 * | data_format    | Data format of the integer operands                                   | DataFormat::Int32/UInt32/UInt16          | True     |
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void lt_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_lt_int<APPROX, data_format>(idst0, idst1, odst)));
}

template <DataFormat data_format>
ALWI void gt_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_gt_int<APPROX, data_format>(idst0, idst1, odst)));
}

template <DataFormat data_format>
ALWI void le_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_le_int<APPROX, data_format>(idst0, idst1, odst)));
}

template <DataFormat data_format>
ALWI void ge_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_ge_int<APPROX, data_format>(idst0, idst1, odst)));
}

/**
 * The following functions initialize the relational operations. They should be invoked prior to calling the execution
 * API. Please refer to execution API documentation (lt_int_tile/gt_int_tile/le_int_tile/ge_int_tile) to find out more
 * about the relational operations.
 */
template <DataFormat data_format>
ALWI void lt_int_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_lt_int_init<data_format>()));
}

template <DataFormat data_format>
ALWI void gt_int_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_gt_int_init<data_format>()));
}

template <DataFormat data_format>
ALWI void le_int_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_le_int_init<data_format>()));
}

template <DataFormat data_format>
ALWI void ge_int_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_ge_int_init<data_format>()));
}

}  // namespace ckernel

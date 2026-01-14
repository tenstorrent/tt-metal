// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_shift.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise shift operation to the left on the input at idst0, by input at idst1: y = x0 << x1
 * Both inputs must be of Int32 or UInt32 or UInt16 data type only. Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void binary_left_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_left_shift<APPROX, data_format>(idst0, idst1, odst)));
}

// clang-format off
/**
 * Performs an elementwise shift operation to the right on the input at idst0, by input at idst1: y = x0 >> x1
 * Both inputs must be of Int32 or UInt32 or UInt16 data type only. Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void binary_right_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_right_shift<APPROX, data_format>(idst0, idst1, odst)));
}

// clang-format off
/**
 * Performs an elementwise logical shift operation to the right on the input at idst0, by input at idst1: y = x0 >> x1
 * Both inputs must be of Int32 data type only. Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void binary_logical_right_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_logical_right_shift<APPROX>(idst0, idst1, odst)));
}

template <bool sign_magnitude_format = false>
ALWI void binary_logical_right_shift_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_logical_right_shift<APPROX, InstrModLoadStore::INT32, sign_magnitude_format>(
        idst0, idst1, odst)));
}

ALWI void binary_logical_right_shift_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_logical_right_shift<APPROX, InstrModLoadStore::INT32>(idst0, idst1, odst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void binary_shift_tile_init() { MATH((llk_math_eltwise_binary_sfpu_shift_init<APPROX>())); }

}  // namespace ckernel

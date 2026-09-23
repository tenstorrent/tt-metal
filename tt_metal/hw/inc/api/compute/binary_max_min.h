// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary_max_min.h"
#endif

// Approach B keeps the per-arch kernel selection here (as the macro version did); approach A moves it into
// ckernel_sfpu_binary_max_min.h.

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise maximum operation on inputs of int32 data type at idst0, idst1: y = max(x0, x1).
 * Output overwrites odst in DST.
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_max_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::BinaryMaxMin<true /* IS_MAX */, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, odst, VectorMode::RC)));
}

/**
 * Please refer to documentation.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_max_int32_tile_init() {
    MATH((sfpu::BinaryMaxMin<true /* IS_MAX */, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

// clang-format off
/**
 * Performs an elementwise maximum operation on inputs of uint32 data type at idst0, idst1: y = max(x0, x1).
 * Output overwrites odst in DST.
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
#ifndef ARCH_QUASAR
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_max_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::BinaryMaxMin<true /* IS_MAX */, DataFormat::UInt32, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, odst, VectorMode::RC)));
}

/**
 * Please refer to documentation.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_max_uint32_tile_init() {
    MATH((sfpu::BinaryMaxMin<true /* IS_MAX */, DataFormat::UInt32, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}
#endif

// clang-format off
/**
 * Performs an elementwise maximum operation on inputs at idst0, idst1: y = max(x0, x1).
 * Output overwrites odst in DST.
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_max_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    MATH((sfpu::BinaryMaxMin<true /* IS_MAX */, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, odst, vector_mode)));
}

/**
 * Please refer to documentation.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_max_tile_init() {
    MATH((sfpu::BinaryMaxMin<true /* IS_MAX */, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

// clang-format off
/**
 * Performs an elementwise minimum operation on inputs of int32 data type at idst0, idst1: y = min(x0, x1).
 * Output overwrites odst in DST.
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_min_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::BinaryMaxMin<false /* IS_MAX */, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, odst, VectorMode::RC)));
}

/**
 * Please refer to documentation.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_min_int32_tile_init() {
    MATH((sfpu::BinaryMaxMin<false /* IS_MAX */, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

// clang-format off
/**
 * Performs an elementwise minimum operation on inputs of uint32 data type at idst0, idst1: y = min(x0, x1).
 * Output overwrites odst in DST.
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
#ifndef ARCH_QUASAR
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_min_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::BinaryMaxMin<false /* IS_MAX */, DataFormat::UInt32, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, odst, VectorMode::RC)));
}

/**
 * Please refer to documentation.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_min_uint32_tile_init() {
    MATH((sfpu::BinaryMaxMin<false /* IS_MAX */, DataFormat::UInt32, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}
#endif

// clang-format off
/**
 * Performs an elementwise minimum operation on inputs at idst0, idst1: y = min(x0, x1).
 * Output overwrites odst in DST.
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_min_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    MATH((sfpu::BinaryMaxMin<false /* IS_MAX */, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, odst, vector_mode)));
}

/**
 * Please refer to documentation.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_min_tile_init() {
    MATH((sfpu::BinaryMaxMin<false /* IS_MAX */, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel

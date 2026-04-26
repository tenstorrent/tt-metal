// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_quant.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise per-tensor affine quantization operation on the first operand using the scaling factor in the second operand.
 * Output overwrites odst in DST.
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
ALWI void quant_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_quant_int32, (APPROX), idst0, idst1, odst, (int)VectorMode::RC)));
}

// clang-format off
/**
 * Performs an elementwise per-tensor affine re-quantization operation on the first operand using the scaling factor in the second operand.
 * Output overwrites odst in DST.
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
ALWI void requant_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_requant_int32, (APPROX), idst0, idst1, odst, (int)VectorMode::RC)));
}

// clang-format off
/**
 * Performs an elementwise per-tensor affine de-quantization operation on the first operand using the scaling factor in the second operand.
 * Output overwrites odst in DST.
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
ALWI void dequant_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_dequant_int32, (APPROX), idst0, idst1, odst, (int)VectorMode::RC)));
}

// clang-format off
/**
 * Initialize the sfpu with the zero point argument of the quantization Op.
 * To be called once at beginning of a kernel.
 *
 * Return value: None
 *
 * | Argument   | Description                           | Data type | Valid range | Required |
 * |------------|---------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void quant_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_CB_ARGS(quant_int32, sfpu::quant_init, (APPROX), zero_point)));
}

// clang-format off
/**
 * Initialize the sfpu with the zero point argument of the re-quantization Op.
 * To be called once at beginning of a kernel.
 *
 * Return value: None
 *
 * | Argument   | Description                              | Data type | Valid range | Required |
 * |------------|------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the re-quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void requant_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_CB_ARGS(requant_int32, sfpu::quant_init, (APPROX), zero_point)));
}

// clang-format off
/**
 * Initialize the sfpu with the zero point argument of the de-quantization Op.
 * To be called once at beginning of a kernel.
 *
 * Return value: None
 *
 * | Argument   | Description                              | Data type | Valid range | Required |
 * |------------|------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the de-quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void dequant_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_CB_ARGS(dequant_int32, sfpu::quant_init, (APPROX), zero_point)));
}

}  // namespace ckernel

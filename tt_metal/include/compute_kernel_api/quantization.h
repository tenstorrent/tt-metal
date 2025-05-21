// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_quant.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise per-tensor affine quantization operation on the first operand using the scaling factor in the second operand.
 * Output overwrites first operand in DST.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void quant_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_quant_int32<APPROX>(idst0, idst1)));
}

// clang-format off
/**
 * Performs an elementwise per-tensor affine re-quantization operation on the first operand using the scaling factor in the second operand.
 * Output overwrites first operand in DST.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void requant_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_requant_int32<APPROX>(idst0, idst1)));
}

// clang-format off
/**
 * Performs an elementwise per-tensor affine de-quantization operation on the first operand using the scaling factor in the second operand.
 * Output overwrites first operand in DST.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void dequant_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_dequant_int32<APPROX>(idst0, idst1)));
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
    MATH((llk_math_eltwise_binary_sfpu_quant_int32_init<APPROX>(zero_point)));
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
    MATH((llk_math_eltwise_binary_sfpu_requant_int32_init<APPROX>(zero_point)));
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
    MATH((llk_math_eltwise_binary_sfpu_dequant_int32_init<APPROX>(zero_point)));
}

}  // namespace ckernel

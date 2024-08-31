// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_cumsum.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void cumsum_row_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_cumsum_row_init<APPROX>() ));
}

/**
 * Performs row-wise cumulative sum (cumsum) of a tile in DST register at index
 * idst_data. The DST register buffer must be in acquired state via
 * *acquire_dst* call. This call is blocking and is only available on the
 * compute engine. Only works for Float16_b and Float32.
 *
 * TODO: fix idst_acc.
 * currently idst_acc is not used and (idst_data + 1) is used for the
 * accumulator.
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst_data      | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_acc       | The index of the tile in DST register buffer to be used as an accumulator  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | first_tile     | Is current tile the first tile?                                            | bool     | 0 to 1                                                | True     |
 * | last_tile      | Is current tile the last tile?                                             | bool     | 0 to 1                                                | True     |
 */
ALWI void cumsum_row_tile(uint32_t idst_data, uint32_t idst_acc, bool first_tile, bool last_tile) {
    MATH(( llk_math_eltwise_unary_sfpu_cumsum_row<APPROX>(idst_data, first_tile, last_tile) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void cumsum_row_int_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_cumsum_row_int_init<APPROX>() ));
}

/**
 * Performs row-wise cumulative sum (cumsum) of a tile in DST register at index
 * idst_data. The DST register buffer must be in acquired state via
 * *acquire_dst* call. This call is blocking and is only available on the
 * compute engine. Only works for Int32.
 *
 * TODO: fix idst_acc.
 * currently idst_acc is not used and (idst_data + 1) is used for the
 * accumulator.
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst_data      | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_acc       | The index of the tile in DST register buffer to be used as an accumulator  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | first_tile     | Is current tile the first tile?                                            | bool     | 0 to 1                                                | True     |
 * | last_tile      | Is current tile the last tile?                                             | bool     | 0 to 1                                                | True     |
 */
ALWI void cumsum_row_int_tile(uint32_t idst_data, uint32_t idst_acc, bool first_tile, bool last_tile) {
    MATH(( llk_math_eltwise_unary_sfpu_cumsum_row_int<APPROX>(idst_data, first_tile, last_tile) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void cumsum_row_flip_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_cumsum_row_flip_init<APPROX>() ));
}

/**
 * Performs row-wise cumulative sum (cumsum) of a tile in DST register at index
 * idst_data in the opposite direction (from bottom to top). The DST register
 * buffer must be in acquired state via *acquire_dst* call. This call is
 * blocking and is only available on the compute engine. Only works for
 * Float16_b and Float32.
 *
 * TODO: fix idst_acc.
 * currently idst_acc is not used and (idst_data + 1) is used for the
 * accumulator.
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst_data      | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_acc       | The index of the tile in DST register buffer to be used as an accumulator  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | mask_h         | Number of rows in the tile except for paddings                             | uint32_t | 1 to TILE_HEIGHT                                      | True     |
 * | first_tile     | Is current tile the first tile?                                            | bool     | 0 to 1                                                | True     |
 * | last_tile      | Is current tile the last tile?                                             | bool     | 0 to 1                                                | True     |
 */
ALWI void cumsum_row_flip_tile(uint32_t idst_data, uint32_t idst_acc, uint32_t mask_h, bool first_tile, bool last_tile) {
    MATH(( llk_math_eltwise_unary_sfpu_cumsum_row_flip<APPROX>(idst_data, mask_h, first_tile, last_tile) ));
}

} // namespace ckernel

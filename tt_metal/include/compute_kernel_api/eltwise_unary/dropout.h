// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "debug/dprint.h"
#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_unary_sfpu_api.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/**
 * Initializes element-wise computation of the dropout operation
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | seed           | The number used to init a PRNG                                             | uint32_t | Greater than or equal to 0                            | False    |
 */
ALWI void dropout_tile_init(uint32_t seed = 0) {
    MATH((llk_math_eltwise_unary_sfpu_dropout_init(seed)));
}

/**
 * Performs element-wise computation of the dropout operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | integer_dropout| Probability value expressed as a 16-bit integer                            | int32_t  | 0 - 65535                                             | False    |
 * | scale_factor   | Scale a tile in DST register. Set value in compliance with bfloat16 format | int32_t  | default value is 0x3f80 (1.0f )                       | False    |
 */
ALWI void dropout_tile(uint32_t idst, int integer_dropout = 32768, int scale_factor = 0x3f80) {
    MATH((llk_math_eltwise_unary_sfpu_dropout(idst, VectorMode::RC, integer_dropout, scale_factor)));
}

} // namespace ckernel

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_bitwise_or.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise bitwise_or computation on input x by y bits , where x is each element of a tile
 * in DST register at index tile_index. The input must be of int data type only. The value is provided as const param0
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     | 
 * | param0          | The value the output is if the input is greater than 0                     | uint32_t |                                                       | True     |
 */
 // clang-format on
ALWI void bitwise_or_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_bitwise_or<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void bitwise_or_tile_init() { MATH((llk_math_eltwise_unary_sfpu_bitwise_or_init<APPROX>())); }

}  // namespace ckernel

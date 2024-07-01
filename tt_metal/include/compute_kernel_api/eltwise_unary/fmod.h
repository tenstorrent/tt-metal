// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_fmod.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

/**
 * Performs element-wise fmod computation on input x by y , where x is each element of a tile
 * in DST register at index tile_index. The input can be of float data type. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                 | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform fmod operation      | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0         | Denominator value to perform fmod operation                                 | uint32_t |                                                       | True     |
 * | param1         | Reciprocal of param0, calculated on-host                                    | uint32_t |                                                       | False    |
 */


ALWI void fmod_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_fmod<APPROX>(idst, param0, param1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void fmod_tile_init() { MATH((llk_math_eltwise_unary_sfpu_fmod_init<APPROX>())); }


} // namespace ckernel

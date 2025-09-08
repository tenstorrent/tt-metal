// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_clamp.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise clamp operation for float. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The min value for the clamp function                                       | uint32_t |                                                       | True     |
 * | param1          | The max value for the clamp function                                       | uint32_t |                                                       | True     |
*/
// clang-format on
ALWI void clamp_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_clamp<APPROX>(idst, param0, param1)));
}

// clang-format off
/**
 * Performs element-wise clamp operation for int32. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The min value for the clamp function                                       | uint32_t |                                                       | True     |
 * | param1          | The max value for the clamp function                                       | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void clamp_tile_int32(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_clamp_int32<APPROX>(idst, param0, param1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void clamp_tile_init() { MATH((llk_math_eltwise_unary_sfpu_clamp_init<APPROX>())); }

}  // namespace ckernel

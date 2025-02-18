// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_prelu.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {
// clang-format off
/**
 * Performs element-wise prelu operation. The value to be prelued in the tile is provided as const param0. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     | 
 * | param0          | Constant value that is being multiplied if the input is lesser than 0      | uint32_t |                                                       | True     |
 */
 // clang-format on
ALWI void prelu_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_prelu<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void prelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_prelu_init<APPROX>())); }

}  // namespace ckernel

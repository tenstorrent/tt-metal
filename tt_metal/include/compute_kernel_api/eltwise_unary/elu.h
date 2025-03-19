// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_elu.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise computation of elu (relu(x) + slope*(exp(x) - 1)*(x <= 0 )) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     | 
 * | slope          | slope used in elu calculation                                              | uint32_t | Greater than 0                                        | True     |
 */
 // clang-format on
ALWI void elu_tile(uint32_t idst, uint32_t param0) { MATH((llk_math_eltwise_unary_sfpu_elu<APPROX>(idst, param0))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void elu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_elu_init<APPROX>())); }

}  // namespace ckernel

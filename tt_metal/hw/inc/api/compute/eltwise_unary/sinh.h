// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sinh.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void sinh_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(sinh, ckernel::sfpu::sinh_init, APPROX)); }

// clang-format off
/**
 * Performs element-wise computation of the hyperbolic sine on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void sinh_tile(uint32_t idst) {
    MATH(SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_sinh, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC));
}

}  // namespace ckernel

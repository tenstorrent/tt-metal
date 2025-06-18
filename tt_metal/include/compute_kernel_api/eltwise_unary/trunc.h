// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_trunc.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {
/**
 * Please refer to documentation for any_init.
 */
ALWI void trunc_tile_init() { MATH((llk_math_eltwise_unary_sfpu_trunc_init<APPROX>())); }

// clang-format off
/**
 * Performs trunc operation on each row of a tile.
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform trunc operation    | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void trunc_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_trunc<APPROX>(idst))); }

// clang-format off
/**
 * Performs trunc operation on each row of a tile.
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform trunc operation    | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void trunc_tile_float32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_trunc_float32<APPROX>(idst))); }

}  // namespace ckernel

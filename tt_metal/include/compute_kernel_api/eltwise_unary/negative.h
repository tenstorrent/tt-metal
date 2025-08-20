/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_negative.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

ALWI void negative_tile_init() { MATH((llk_math_eltwise_unary_sfpu_negative_init<APPROX>())); }
// clang-format off
/**
 * Performs element-wise computation of the negative on each element of a tile
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
ALWI void negative_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_negative<APPROX>(idst))); }

ALWI void negative_tile_int32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_negative_int<APPROX>(idst))); }

}  // namespace ckernel

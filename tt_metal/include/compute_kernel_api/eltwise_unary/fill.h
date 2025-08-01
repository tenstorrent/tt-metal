// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_fill.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise fill operation. The value to be filled in the tile is provided as const param0. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value the output is if the input is greater than 0                     | float    |                                                       | True     |
 */
// clang-format on
ALWI void fill_tile(uint32_t idst, float param0) { MATH((llk_math_eltwise_unary_sfpu_fill<APPROX>(idst, param0))); }

ALWI void fill_tile_int(uint32_t idst, uint param0) {
    MATH((llk_math_eltwise_unary_sfpu_fill_int<APPROX>(idst, param0)));
}

// clang-format off
/**
 * Performs element-wise fill operation. The value to be filled in the tile is provided as const param0, which is
 * interpreted as a bit-cast representation of a floating-point value. The DST register buffer must be in acquired
 * state via *acquire_dst* call. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The bit-cast representation of a floating-point value to be used as output | uint32_t | Must represent a valid bit-cast float value           | True     |
 */
// clang-format on
ALWI void fill_tile_bitcast(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_fill_bitcast<APPROX>(idst, param0)));
}
/**
 * Please refer to documentation for any_init.
 */
ALWI void fill_tile_init() { MATH((llk_math_eltwise_unary_sfpu_fill_init<APPROX>())); }

}  // namespace ckernel

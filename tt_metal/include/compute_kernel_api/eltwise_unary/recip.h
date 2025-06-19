// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_recip.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void recip_tile_init() { MATH((llk_math_eltwise_unary_sfpu_reciprocal_init<APPROX>())); }
// clang-format off
/**
 * Performs element-wise computation of the reciprocal on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 * Only works for Float32, Float16_b, Bfp8_b data formats for full accuracy.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | vector_mode | Specifies the vector mode for computation (e.g., Row, Column). (default: VectorMode::RC) | int      | Subject to specific hardware/kernel limits          | False    |
 */
// clang-format on
ALWI void recip_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC) {
    MATH((llk_math_eltwise_unary_sfpu_reciprocal<APPROX, DST_ACCUM_MODE>(idst, vector_mode)));
}

}  // namespace ckernel

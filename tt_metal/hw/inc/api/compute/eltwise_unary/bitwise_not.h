// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_bitwise_not.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise bitwise_not computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to modify the computation of  | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void bitwise_not_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_bitwise_not, (APPROX), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void bitwise_not_tile_init() { MATH(SFPU_UNARY_INIT(bitwise_not)); }

}  // namespace ckernel

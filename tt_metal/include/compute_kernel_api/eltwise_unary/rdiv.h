// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_rdiv.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void rdiv_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rdiv_init<APPROX>())); }

// clang-format off
/**
 * Performs element-wise computation of reciprocal divide on each element of a
 * tile in DST register at index idst. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | value          | The numerator value to divide by each element of the tile                  | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <RoundingMode rounding_mode = RoundingMode::None>
ALWI void rdiv_tile(uint32_t dst_index, uint32_t value, int vector_mode = (int)VectorMode::RC) {
    MATH((llk_math_eltwise_unary_sfpu_rdiv<APPROX, DST_ACCUM_MODE, rounding_mode>(dst_index, value, vector_mode)));
}

}  // namespace ckernel

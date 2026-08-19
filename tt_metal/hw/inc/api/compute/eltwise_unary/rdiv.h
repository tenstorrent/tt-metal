// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_rdiv.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void rdiv_tile_init() { MATH(SFPU_UNARY_INIT_FN(rdiv, sfpu::rdiv_init, (APPROX))); }

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
ALWI void rdiv_tile(uint32_t dst_index, uint32_t value, VectorMode vector_mode = VectorMode::RC) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rdiv,
        (APPROX, DST_ACCUM_MODE, rounding_mode, 8 /* ITERATIONS */),
        dst_index,
        vector_mode,
        value));
}

}  // namespace ckernel

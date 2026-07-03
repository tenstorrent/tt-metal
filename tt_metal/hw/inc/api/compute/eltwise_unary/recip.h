// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {
/**
 * Please refer to documentation for any_init.
 */
template <bool legacy_compat = true>
ALWI void recip_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(reciprocal, sfpu::recip_init, (APPROX, DST_ACCUM_MODE, legacy_compat)));
}
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
 * | vector_mode | Specifies the vector mode for computation (e.g., Row, Column). (default: VectorMode::RC) | VectorMode | Subject to specific hardware/kernel limits          | False    |
 */
// clang-format on
template <bool legacy_compat = true>
ALWI void recip_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_reciprocal,
        (APPROX, DST_ACCUM_MODE, 8 /*ITERATIONS*/, legacy_compat),
        idst,
        vector_mode));
}
}  // namespace ckernel

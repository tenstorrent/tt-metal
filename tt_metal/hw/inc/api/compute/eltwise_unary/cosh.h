// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api.h"

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
template <bool approx = true>
ALWI void cosh_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_cosh_init<approx>()));
}

// clang-format off
/**
 * Performs element-wise computation of hyperbolic cosine on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool approx = true>
ALWI void cosh_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_cosh<approx, DST_ACCUM_MODE>(idst)));
}

}  // namespace ckernel

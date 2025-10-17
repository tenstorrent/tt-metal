// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_hardgelu.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise computation of hardgelu(x) = x * (x + 2.8).clamp(0.0, 5.0) / 5 on each element of a tile
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
ALWI void hardgelu_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_hardgelu<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardgelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_hardgelu_init<APPROX>())); }

}  // namespace ckernel

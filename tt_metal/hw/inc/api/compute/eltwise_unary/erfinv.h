// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_erfinv.h"
#endif

namespace ckernel {
// clang-format off
/**
 * Performs element-wise computation of the inverse of the error function on each element of a tile
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
ALWI void erfinv_tile(std::uint32_t idst) { MATH((sfpu::Erfinv<APPROX>::run(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void erfinv_tile_init() { MATH((sfpu::Erfinv<APPROX>::init())); }
}  // namespace ckernel

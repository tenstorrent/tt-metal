// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#endif

namespace ckernel {
/**
 * Please refer to documentation for any_init.
 */
ALWI void sqrt_tile_init() { MATH((sfpu::Sqrt<APPROX>::init())); }

// clang-format off
/**
 * Performs element-wise computation of the square root on each element of a
 * tile in DST register at index idst. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool FAST_APPROX = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sqrt_tile(std::uint32_t idst) {
    MATH((sfpu::Sqrt<APPROX, 8 /* ITERATIONS */, is_fp32_dest_acc_en, FAST_APPROX>::run(idst)));
}

}  // namespace ckernel

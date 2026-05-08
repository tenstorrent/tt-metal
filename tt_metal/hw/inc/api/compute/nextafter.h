// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_nextafter.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise nextafter operation: y = nextafter(x0, x1).
 * Output overwrites odst in DST. Matches torch.nextafter / IEEE-754 semantics
 * (NaN propagates, x0 == x1 returns x1, x0 == 0 returns the smallest subnormal
 * with sign(x1)).
 *
 * Inputs must be float32 or bfloat16; ``data_format`` selects the in-DST
 * representation so the integer ULP step matches the underlying type
 * (1 ULP for fp32, 0x10000 for bfloat16).
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument  | Description                                                           | Type     | Valid Range                                           | Required |
 * |-----------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0     | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1     | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst      | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format = DataFormat::Float32>
ALWI void nextafter_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_nextafter<APPROX, data_format, 8>(idst0, idst1, odst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void nextafter_binary_tile_init() { MATH((llk_math_eltwise_binary_sfpu_nextafter_init<APPROX>())); }

}  // namespace ckernel

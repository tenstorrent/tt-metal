// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_unary_shift.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise left_shift computation on input x by param0 bits, where x is each element of a tile
 * in DST register at index idst. The input must be of integer data type: Int32, UInt32, or UInt16. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * A shift amount outside [0, 31] produces 0 for every element.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The number of bits to shift the input by                                   | uint32_t |                                                       | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void left_shift_tile(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::LeftShift<APPROX, data_format>::run(idst, param0)));
}

// clang-format off
/**
 * Performs element-wise right_shift computation on input x by param0 bits, where x is each element of a tile in DST
 * register at index idst. The input must be of integer data type: Int32, UInt32, or UInt16. Int32 uses an arithmetic
 * shift (the sign bit is replicated into the vacated high bits). UInt32 and UInt16 use a logical shift (zeros fill the
 * vacated high bits). The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine.
 *
 * A shift amount >= 32 saturates to 31. For Int32 that yields 0 for non-negative inputs and -1 for negative inputs; for
 * UInt32 and UInt16 it yields a logical shift by 31.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The number of bits to shift the input by                                   | uint32_t |                                                       | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void right_shift_tile(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::RightShift<APPROX, data_format>::run(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void left_shift_tile_init() { MATH((sfpu::LeftShift<APPROX>::init())); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void right_shift_tile_init() { MATH((sfpu::RightShift<APPROX>::init())); }

}  // namespace ckernel

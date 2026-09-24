// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_remainder.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise remainder computation on input x by y , where x is each element of a tile
 * in DST register at index tile_index. The input can be of float data type. The denominator is provided to
 * remainder_tile_init and loaded into the SFPU constant registers. The
 * DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on
 * the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                 | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform remainder operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void remainder_tile(std::uint32_t idst) { MATH((sfpu::Remainder<APPROX>::run(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void remainder_tile_init(std::uint32_t param0, std::uint32_t param1) {
    MATH((sfpu::Remainder<APPROX>::init(param0, param1)));
}

// clang-format off
/**
 * Performs element-wise unsigned remainder computation on input x by the uint32 scalar divisor,
 * where x is each element of a tile in DST register at index tile_index. The result is x mod divisor
 * in [0, divisor). The DST register buffer must be in acquired state via *acquire_dst* call. This
 * call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                 | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform remainder operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0         | The unsigned divisor                                                        | uint32_t | [1, 4294967295]                                       | True     |
 */
// clang-format on
ALWI void remainder_tile_uint32(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::RemainderUint32Scalar<APPROX>::run(idst, param0 /*divisor*/)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void remainder_tile_uint32_init() { MATH((sfpu::RemainderUint32Scalar<APPROX>::init())); }

}  // namespace ckernel

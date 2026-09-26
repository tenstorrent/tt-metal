// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_where.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise where operation with the three inputs: y = where(x0,x1,x2)
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 *
 * @tparam data_format Template argument specifying the data type.
 * Supported data formats are: DataFormat::Int32, DataFormat::UInt32, DataFormat::UInt16
 *
 * | Argument              | Description                                                              | Type     | Valid Range                                           | Required |
 * |-----------------------|--------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as condition operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as first operand     | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst2                 | The index of the tile in DST register buffer to use as second operand    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst                  | The index of the tile in DST register buffer to use as output            | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void where_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t idst2, std::uint32_t odst) {
    MATH((sfpu::Where<APPROX, data_format>::run(idst0, idst1, idst2, odst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void where_tile_init() { MATH((sfpu::Where<APPROX>::init())); }

}  // namespace ckernel

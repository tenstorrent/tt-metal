// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
// The integer add kernel and its op class live in ckernel_sfpu_add_int.h on Wormhole/Blackhole and in
// ckernel_sfpu_add.h on Quasar.
#ifdef ARCH_QUASAR
#include "ckernel_sfpu_add.h"
#else
#include "ckernel_sfpu_add_int.h"
#endif
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise add operation with the two integer inputs: y = add(x0,x1)
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * @tparam data_format Template argument specifying the data type.
 * Supported data formats are: DataFormat::Int32, DataFormat::UInt32, DataFormat::UInt16\n
 *
 * Return value: None
 *
 * | Argument | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0    | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1    | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst     | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void add_int_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::AddInt<APPROX, data_format>::run(idst0, idst1, odst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void add_int_tile_init() { MATH((sfpu::AddInt<APPROX>::init())); }

}  // namespace ckernel

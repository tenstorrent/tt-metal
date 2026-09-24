// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_fill.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise fill operation. The value to be filled in the tile is provided as const param0. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Value to fill tile with.                                                   | float    |                                                       | True     |
 */
// clang-format on
ALWI void fill_tile(std::uint32_t idst, float param0) { MATH((sfpu::Fill<APPROX>::run(idst, param0))); }

// clang-format off
/**
 * Performs element-wise fill operation. The value to be filled in the tile is provided as const param0. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * @tparam data_format Template argument specifying the data type.
 * Supported data formats are: DataFormat::Int32, DataFormat::UInt32, DataFormat::UInt16.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Value to fill tile with (unsigned integer)                                 | uint32_t |                                                       | True     |
 */
template <DataFormat DATA_FORMAT>
ALWI void fill_tile_int(std::uint32_t idst, std::uint32_t param0) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for fill_tile_int. Supported: Int32, UInt32, UInt16");
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        (DATA_FORMAT == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
    MATH((sfpu::FillInt<APPROX, INSTRUCTION_MODE>::run(idst, param0)));
}

// clang-format off
/**
 * Performs element-wise fill operation. The value to be filled in the tile is provided as const param0, which is
 * interpreted as a bit-cast representation of a floating-point value. The DST register buffer must be in acquired
 * state via *acquire_dst* call. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The bit-cast representation of a floating-point value to be used as output | uint32_t | Must represent a valid bit-cast float value           | True     |
 */
// clang-format on
ALWI void fill_tile_bitcast(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::FillBitcast<APPROX>::run(idst, param0)));
}
/**
 * Please refer to documentation for any_init.
 */
ALWI void fill_tile_init() { MATH((sfpu::Fill<APPROX>::init())); }

}  // namespace ckernel

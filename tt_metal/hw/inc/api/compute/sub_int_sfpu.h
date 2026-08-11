// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "sfpu/ckernel_sfpu_sub_int.h"
#include "ckernel_sfpu_rsub_int32.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise sub operation with the two integer inputs: y = sub(x0,x1)
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * @tparam data_format Template argument specifying the data type.
 * Supported data formats are: DataFormat::Int32, DataFormat::UInt32, DataFormat::UInt16
 *
 * Return value: None
 *
 * | Argument              | Description                                                           | Type     | Valid Range                                           | Required |
 * |-----------------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst                  | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void sub_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for sub_int. Supported data formats are: Int32, UInt32, UInt16");
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        (data_format == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _sub_int_,
        (APPROX, 8 /* ITERATIONS */, INSTRUCTION_MODE, false /* SIGN_MAGNITUDE_FORMAT */),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

// clang-format off
/**
 * Performs an elementwise rsub operation with the two integer inputs: x = rsub(x0,x1) = x1 - x0.
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * @tparam data_format Template argument specifying the data type.
 * Supported data formats are: DataFormat::Int32, DataFormat::UInt32, DataFormat::UInt16
 *
 * Return value: None
 *
 * | Argument              | Description                                                           | Type     | Valid Range                                           | Required |
 * |-----------------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst                  | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void rsub_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for rsub_int. Supported data formats are: Int32, UInt32, UInt16");
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        (data_format == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsub_int,
        (APPROX, INSTRUCTION_MODE, 8 /* ITERATIONS */),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}
/**
 * Please refer to documentation for any_init.
 */
ALWI void sub_int_tile_init() { MATH((SFPU_BINARY_INIT(unused))); }

ALWI void rsub_int_tile_init() { MATH((SFPU_BINARY_INIT(unused))); }

}  // namespace ckernel

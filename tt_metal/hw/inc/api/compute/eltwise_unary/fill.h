// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_fill.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
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
ALWI void fill_tile(uint32_t idst, float param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_fill_, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC, param0));
}

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
ALWI void fill_tile_int(uint32_t idst, uint32_t param0) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || is_uint32_format(DATA_FORMAT) || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for fill_tile_int. Supported: Int32, UInt32, UInt16");
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        (DATA_FORMAT == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_fill_int_,
        (APPROX, INSTRUCTION_MODE, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0));
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
ALWI void fill_tile_bitcast(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_fill_bitcast_,
        (APPROX, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0));
}
// clang-format off
/**
 * Fills rows [row_begin, row_end) of the 32x32 tile in DST with the given value and leaves every other row as it is:
 * a 4-row band that lies inside the range is written with whole-vector stores, a band cut by row_begin or row_end with
 * a lane-masked store. The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine. Call *fill_tile_init* first.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | row_begin       | The first tile row to fill                                                 | uint32_t | 0 to 31, less than row_end                            | True     |
 * | row_end         | One past the last tile row to fill                                         | uint32_t | row_begin + 1 to 32                                   | True     |
 * | value           | Value to fill the rows with                                                | float    |                                                       | True     |
 */
// clang-format on
ALWI void fill_rows_tile(uint32_t idst, uint32_t row_begin, uint32_t row_end, float value) {
    MATH(LLK_ASSERT(row_begin < row_end && row_end <= 32, "fill_rows_tile: need 0 <= row_begin < row_end <= 32"));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_fill_rows_,
        (APPROX),
        idst,
        VectorMode::None,
        row_begin,
        row_end,
        value));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void fill_tile_init() { MATH(SFPU_UNARY_INIT(fill)); }

}  // namespace ckernel

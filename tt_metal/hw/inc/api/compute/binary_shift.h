// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_shift.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise shift operation to the left on the input at idst0, by input at idst1: y = x0 << x1
 * Both inputs must be of same data type only. Output overwrites odst in DST.
 *
 * A shift amount < 0 or >= 32 produces 0.
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
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_left_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for left shift. Supported data formats are: Int32, UInt32, UInt16");
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        (data_format == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
    MATH((sfpu::BinaryShift<APPROX, sfpu::BinaryShiftOp::LEFT, INSTRUCTION_MODE, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst0, idst1, odst, VectorMode::RC)));
}

// clang-format off
/**
 * Performs an elementwise shift operation to the right on the input at idst0, by input at idst1: y = x0 >> x1
 * Both inputs must be of same data type only. Output overwrites odst in DST.
 *
 * Int32 uses an arithmetic shift. UInt32 uses a logical shift. UInt16 uses the Int32 path.
 *
 * For UInt32 a shift amount >= 32 saturates to 31, matching scalar `right_shift_tile`. For Int32 and UInt16 a shift
 * amount < 0 or >= 32 produces 0.
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
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_right_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for right shift. Supported data formats are: Int32, UInt32, UInt16");
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        (data_format == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
    // UInt32 uses a logical shift and clamps counts >= 32 to 31, matching the
    // scalar right-shift contract. UInt16 and Int32 retain their existing paths.
    MATH(
        (sfpu::BinaryShift < APPROX,
         (data_format == DataFormat::UInt32) ? sfpu::BinaryShiftOp::CLAMPED_LOGICAL_RIGHT : sfpu::BinaryShiftOp::RIGHT,
         INSTRUCTION_MODE,
         DST_SYNC_MODE,
         is_fp32_dest_acc_en > ::calculate(idst0, idst1, odst, VectorMode::RC)));
}

// clang-format off
/**
 * Performs an elementwise logical shift operation to the right on the input at idst0, by input at idst1: y = x0 >> x1
 * Both inputs must be same data type only. Vacated high bits are filled with zeros. Output overwrites odst in DST.
 *
 * A shift amount < 0 or >= 32 produces 0. Unlike `binary_right_shift_tile` on UInt32, out-of-range counts are not
 * saturated to 31.
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
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_logical_right_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for logical right shift. Supported data formats are: Int32, UInt32, UInt16");
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        (data_format == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
    MATH((sfpu::BinaryShift<
          APPROX,
          sfpu::BinaryShiftOp::LOGICAL_RIGHT,
          INSTRUCTION_MODE,
          DST_SYNC_MODE,
          is_fp32_dest_acc_en>::calculate(idst0, idst1, odst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binary_shift_tile_init() {
    // The init is shared by all shift variants and data formats (shared SFPU init only).
    MATH((sfpu::BinaryShift<
          APPROX,
          sfpu::BinaryShiftOp::LEFT,
          InstrModLoadStore::INT32,
          DST_SYNC_MODE,
          is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel

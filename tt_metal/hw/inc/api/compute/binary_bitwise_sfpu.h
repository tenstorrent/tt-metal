// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary_bitwise.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise binary bitwise operation with the two inputs: y = bitwise(x0,x1)
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
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on

namespace detail {
template <DataFormat data_format>
constexpr InstrModLoadStore bitwise_instr_mode() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for bitwise operation. Supported data formats are: Int32, UInt32, UInt16");
    return data_format == DataFormat::UInt16 ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
}
}  // namespace detail

template <DataFormat data_format>
ALWI void bitwise_and_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = detail::bitwise_instr_mode<data_format>();
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_bitwise,
        (APPROX, sfpu::BinaryBitwiseOp::AND, INSTRUCTION_MODE),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

template <DataFormat data_format>
ALWI void bitwise_or_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = detail::bitwise_instr_mode<data_format>();
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_bitwise,
        (APPROX, sfpu::BinaryBitwiseOp::OR, INSTRUCTION_MODE),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

template <DataFormat data_format>
ALWI void bitwise_xor_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = detail::bitwise_instr_mode<data_format>();
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_bitwise,
        (APPROX, sfpu::BinaryBitwiseOp::XOR, INSTRUCTION_MODE),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void binary_bitwise_tile_init() { MATH((SFPU_BINARY_INIT(unused))); }

}  // namespace ckernel

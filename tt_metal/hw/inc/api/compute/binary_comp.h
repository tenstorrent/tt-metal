// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary_comp.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise comparison operation with two integer inputs: y = comparison_op(x0,x1)
 * Supports Int32, UInt32 and UInt16 data formats (selected via the data_format template parameter).
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * Return value: None
 *
 * | Template Param | Description                                                           | Valid Values                             | Required |
 * |----------------|-----------------------------------------------------------------------|------------------------------------------|----------|
 * | data_format    | Data format of the integer operands                                   | DataFormat::Int32/UInt32/UInt16          | True     |
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on

// eq/ne/lt/le/ge on integers do not exist on Quasar; only gt_int does.
#ifndef ARCH_QUASAR
template <DataFormat data_format>
ALWI void eq_int_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::BinaryCompInt<APPROX, sfpu::CompareOp::eq, data_format>::run(idst0, idst1, odst)));
}

template <DataFormat data_format>
ALWI void ne_int_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::BinaryCompInt<APPROX, sfpu::CompareOp::ne, data_format>::run(idst0, idst1, odst)));
}

template <DataFormat data_format>
ALWI void lt_int_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::BinaryCompInt<APPROX, sfpu::CompareOp::lt, data_format>::run(idst0, idst1, odst)));
}
#endif

template <DataFormat data_format>
ALWI void gt_int_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::BinaryCompInt<APPROX, sfpu::CompareOp::gt, data_format>::run(idst0, idst1, odst)));
}

// eq/ne/lt/le/ge on integers do not exist on Quasar; only gt_int does.
#ifndef ARCH_QUASAR
template <DataFormat data_format>
ALWI void le_int_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::BinaryCompInt<APPROX, sfpu::CompareOp::le, data_format>::run(idst0, idst1, odst)));
}

template <DataFormat data_format>
ALWI void ge_int_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::BinaryCompInt<APPROX, sfpu::CompareOp::ge, data_format>::run(idst0, idst1, odst)));
}
#endif

/**
 * The following functions initialize the relational operations. They should be invoked prior to calling the execution
 * API. Please refer to execution API documentation to find out more about the relational operations.
 */
// eq/ne/lt/le/ge on integers do not exist on Quasar; only gt_int does.
#ifndef ARCH_QUASAR
template <DataFormat data_format>
ALWI void eq_int_tile_init() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for eq_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((sfpu::BinaryCompInt<APPROX, sfpu::CompareOp::eq, data_format>::init()));
}

template <DataFormat data_format>
ALWI void ne_int_tile_init() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for ne_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((sfpu::BinaryCompInt<APPROX, sfpu::CompareOp::ne, data_format>::init()));
}

template <DataFormat data_format>
ALWI void lt_int_tile_init() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for lt_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((sfpu::BinaryCompInt<APPROX, sfpu::CompareOp::lt, data_format>::init()));
}
#endif

template <DataFormat data_format>
ALWI void gt_int_tile_init() {
    // BinaryCompInt checks data_format against what each architecture supports.
    MATH((sfpu::BinaryCompInt<APPROX, sfpu::CompareOp::gt, data_format>::init()));
}

// eq/ne/lt/le/ge on integers do not exist on Quasar; only gt_int does.
#ifndef ARCH_QUASAR
template <DataFormat data_format>
ALWI void le_int_tile_init() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for le_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((sfpu::BinaryCompInt<APPROX, sfpu::CompareOp::le, data_format>::init()));
}

template <DataFormat data_format>
ALWI void ge_int_tile_init() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for ge_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((sfpu::BinaryCompInt<APPROX, sfpu::CompareOp::ge, data_format>::init()));
}
#endif

}  // namespace ckernel

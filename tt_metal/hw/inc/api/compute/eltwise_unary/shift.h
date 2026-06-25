// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_unary_shift.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise left_shift computation on input x by param0 bits, where x is each element of a tile
 * in DST register at index idst. The input must be of int data type only. The DST register buffer must be in acquired
 * state via *acquire_dst* call. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The number of bits to shift the input by                                   | uint32_t |                                                       | True     |
 */
// clang-format on
namespace detail {
template <DataFormat data_format>
constexpr InstrModLoadStore unary_shift_instr_mode() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for shift operation. Supported data formats are: Int32, UInt32, UInt16");
    return data_format == DataFormat::UInt16 ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
}
}  // namespace detail

template <DataFormat data_format>
ALWI void left_shift_tile(uint32_t idst, uint32_t param0) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = detail::unary_shift_instr_mode<data_format>();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_left_shift, (APPROX, INSTRUCTION_MODE), idst, VectorMode::RC, param0));
}

// clang-format off
/**
 * Performs element-wise (arithmetic) right_shift computation on input x by param0 bits, where x is each element of a
 * tile in DST register at index idst. The input must be of int data type only. The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
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
ALWI void right_shift_tile(uint32_t idst, uint32_t param0) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = detail::unary_shift_instr_mode<data_format>();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_right_shift,
        (APPROX, INSTRUCTION_MODE),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void left_shift_tile_init() { MATH(SFPU_UNARY_INIT(left_shift)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void right_shift_tile_init() { MATH(SFPU_UNARY_INIT(right_shift)); }

}  // namespace ckernel

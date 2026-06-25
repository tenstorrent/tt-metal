// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_bitwise.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an element-wise bitwise operation between each element of a tile in the DST register at
 * index idst and an immediate scalar param0: y = bitwise(x, param0). The input must be of int data
 * type only. Output overwrites the input tile in DST. The DST register buffer must be in acquired state
 * via *acquire_dst* call. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The scalar second operand of the bitwise operation                         | uint32_t |                                                       | True     |
 */
// clang-format on

namespace detail {
template <DataFormat data_format>
constexpr InstrModLoadStore unary_bitwise_instr_mode() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for bitwise operation. Supported data formats are: Int32, UInt32, UInt16");
    return data_format == DataFormat::UInt16 ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
}
}  // namespace detail

template <DataFormat data_format>
ALWI void bitwise_and_tile(uint32_t idst, uint32_t param0) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = detail::unary_bitwise_instr_mode<data_format>();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_unary_bitwise,
        (APPROX, sfpu::UnaryBitwiseOp::AND, INSTRUCTION_MODE),
        idst,
        VectorMode::RC,
        param0));
}

template <DataFormat data_format>
ALWI void bitwise_or_tile(uint32_t idst, uint32_t param0) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = detail::unary_bitwise_instr_mode<data_format>();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_unary_bitwise,
        (APPROX, sfpu::UnaryBitwiseOp::OR, INSTRUCTION_MODE),
        idst,
        VectorMode::RC,
        param0));
}

template <DataFormat data_format>
ALWI void bitwise_xor_tile(uint32_t idst, uint32_t param0) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = detail::unary_bitwise_instr_mode<data_format>();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_unary_bitwise,
        (APPROX, sfpu::UnaryBitwiseOp::XOR, INSTRUCTION_MODE),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void bitwise_and_tile_init() { MATH(SFPU_UNARY_INIT(bitwise_and)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void bitwise_or_tile_init() { MATH(SFPU_UNARY_INIT(bitwise_or)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void bitwise_xor_tile_init() { MATH(SFPU_UNARY_INIT(bitwise_xor)); }

}  // namespace ckernel

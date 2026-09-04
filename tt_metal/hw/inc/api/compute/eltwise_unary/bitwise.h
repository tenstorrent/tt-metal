// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_bitwise.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an element-wise bitwise AND between each element of a tile in the DST register at index idst and an
 * immediate scalar param0: y = x & param0. The input must be of integer data type: Int32, UInt32, or UInt16. Output
 * overwrites the input tile in DST. The DST register buffer must be in acquired state via *acquire_dst* call. This
 * call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The scalar second operand of the bitwise operation                         | uint32_t |                                                       | True     |
 */
// clang-format on
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void bitwise_and_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::Bitwise<APPROX, sfpu::UnaryBitwiseOp::AND, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
}

// clang-format off
/**
 * Performs an element-wise bitwise OR between each element of a tile in the DST register at index idst and an
 * immediate scalar param0: y = x | param0. The input must be of integer data type: Int32, UInt32, or UInt16. Output
 * overwrites the input tile in DST. The DST register buffer must be in acquired state via *acquire_dst* call. This
 * call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The scalar second operand of the bitwise operation                         | uint32_t |                                                       | True     |
 */
// clang-format on
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void bitwise_or_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::Bitwise<APPROX, sfpu::UnaryBitwiseOp::OR, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
}

// clang-format off
/**
 * Performs an element-wise bitwise XOR between each element of a tile in the DST register at index idst and an
 * immediate scalar param0: y = x ^ param0. The input must be of integer data type: Int32, UInt32, or UInt16. Output
 * overwrites the input tile in DST. The DST register buffer must be in acquired state via *acquire_dst* call. This
 * call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The scalar second operand of the bitwise operation                         | uint32_t |                                                       | True     |
 */
// clang-format on
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void bitwise_xor_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::Bitwise<APPROX, sfpu::UnaryBitwiseOp::XOR, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void bitwise_and_tile_init() {
    // The init is the shared SFPU init; the data-format parameter of the op struct is irrelevant here.
    MATH((sfpu::Bitwise<APPROX, sfpu::UnaryBitwiseOp::AND, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void bitwise_or_tile_init() {
    // The init is the shared SFPU init; the data-format parameter of the op struct is irrelevant here.
    MATH((sfpu::Bitwise<APPROX, sfpu::UnaryBitwiseOp::OR, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void bitwise_xor_tile_init() {
    // The init is the shared SFPU init; the data-format parameter of the op struct is irrelevant here.
    MATH((sfpu::Bitwise<APPROX, sfpu::UnaryBitwiseOp::XOR, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}

}  // namespace ckernel

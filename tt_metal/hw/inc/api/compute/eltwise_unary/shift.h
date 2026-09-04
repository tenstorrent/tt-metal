// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_unary_shift.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise left_shift computation on input x by param0 bits, where x is each element of a tile
 * in DST register at index idst. The input must be of integer data type: Int32, UInt32, or UInt16. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * A shift amount outside [0, 31] produces 0 for every element.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The number of bits to shift the input by                                   | uint32_t |                                                       | True     |
 */
// clang-format on
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void left_shift_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryShift<APPROX, true /*IS_LEFT_SHIFT*/, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
}

// clang-format off
/**
 * Performs element-wise (arithmetic) right_shift computation on input x by param0 bits, where x is each element of a
 * tile in DST register at index idst. The input must be of integer data type: Int32, UInt32, or UInt16. The shift is
 * arithmetic: the sign bit is replicated into the vacated high bits (negative inputs shift in 1s). The DST register
 * buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the compute
 * engine.
 *
 * A shift amount outside [0, 31] produces 0 for non-negative inputs and -1 for negative inputs.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The number of bits to shift the input by                                   | uint32_t |                                                       | True     |
 */
// clang-format on
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void right_shift_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryShift<APPROX, false /*IS_LEFT_SHIFT*/, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void left_shift_tile_init() {
    // The init is the shared SFPU init; the data-format parameter of the op struct is irrelevant here.
    MATH((sfpu::UnaryShift<APPROX, true /*IS_LEFT_SHIFT*/, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void right_shift_tile_init() {
    MATH((sfpu::UnaryShift<APPROX, false /*IS_LEFT_SHIFT*/, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}

}  // namespace ckernel

// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_typecast.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise typecast operation on the input.
 * Supports following typecasts:
 *  Float16_b <-> Float32
 *  Float16_b <-> Int32
 *  Float16_b <-> UInt16
 *  Float16_b <-> UInt32
 *  Float16_b <-> UInt8
 *  Float32 <-> Int32
 *  Float32 <-> UInt16
 *  Float32 <-> UInt32
 *  Float32 <-> UInt8
 *  Bfp8_b <-> Int32
 *  Bfp8_b <-> UInt16
 *  Bfp8_b <-> UInt32
 *  Bfp8_b <-> UInt8
 *  Bfp8_b <-> Float16_b
 *  Bfp8_b <-> Float32
 *  Bfp4_b <-> Int32
 *  Bfp4_b <-> UInt16
 *  Bfp4_b <-> UInt32
 *  Bfp4_b <-> UInt8
 *  Bfp4_b <-> Bfp8_b
 *  Bfp4_b <-> Float16_b
 *  Bfp4_b <-> Float32
 *  UInt16 <-> UInt32
 *  UInt16 <-> Int32
 *  UInt16 <-> UInt8
 *
 * For input/output to be UInt32, Int32, or Float32, Dest must be in 32 bit mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform typecast operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | IN_DTYPE       | Input data format                                                          | uint32_t | Must be valid tt::DataFormat                          | True     |
 * | OUT_DTYPE      | Desired output data format                                                 | uint32_t | Must be valid tt::DataFormat                          | True     |
 */
// clang-format on
template <uint32_t IN_DTYPE, uint32_t OUT_DTYPE, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void typecast_tile(uint32_t idst) {
    MATH((sfpu::Typecast<
          static_cast<DataFormat>(IN_DTYPE),
          static_cast<DataFormat>(OUT_DTYPE),
          APPROX,
          DST_SYNC_MODE,
          is_fp32_dest_acc_en>::calculate(idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <uint32_t IN_DTYPE, uint32_t OUT_DTYPE, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void typecast_tile_init() {
    MATH((sfpu::Typecast<
          static_cast<DataFormat>(IN_DTYPE),
          static_cast<DataFormat>(OUT_DTYPE),
          APPROX,
          DST_SYNC_MODE,
          is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel

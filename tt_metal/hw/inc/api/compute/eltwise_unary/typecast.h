// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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
 *  Float16_b <-> Int8
 *  Float32 <-> Int32
 *  Float32 <-> UInt16
 *  Float32 <-> UInt32
 *  Float32 <-> UInt8
 *  Float32 <-> Int8
 *  Bfp8_b <-> Int32
 *  Bfp8_b <-> UInt16
 *  Bfp8_b <-> UInt32
 *  Bfp8_b <-> UInt8
 *  Bfp8_b <-> Int8
 *  Bfp8_b <-> Float16_b
 *  Bfp8_b <-> Float32
 *  Bfp4_b <-> Int32
 *  Bfp4_b <-> UInt16
 *  Bfp4_b <-> UInt32
 *  Bfp4_b <-> UInt8
 *  Bfp4_b <-> Int8
 *  Bfp4_b <-> Bfp8_b
 *  Bfp4_b <-> Float16_b
 *  Bfp4_b <-> Float32
 *  UInt16 <-> UInt32
 *  UInt16 <-> Int32
 *  UInt16 <-> UInt8
 *  UInt16 <-> Int8
 *  Int32 <-> Int8
 *  UInt32 <-> Int8
 *  UInt8 <-> Int8
 *
 * For input/output to be UInt32, Int32, or Float32, Dest must be in 32 bit mode.
 *
 * For input/output to be Int8, the caller must additionally declare the circular buffers as UInt8 instead
 * of Int8, so the raw 2's complement byte is zero-extended instead of being decoded as sign-magnitude,
 * and must put Dest in 32 bit mode, which is enforced by a static_assert. The kernels below do the sign
 * handling themselves on that raw byte.
 * Int8 is not available on Quasar.
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
template <std::uint32_t IN_DTYPE, std::uint32_t OUT_DTYPE, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void typecast_tile(std::uint32_t idst) {
#ifdef TRISC_MATH
    constexpr DataFormat in_format = static_cast<DataFormat>(IN_DTYPE);
    constexpr DataFormat out_format = static_cast<DataFormat>(OUT_DTYPE);
    using Op = sfpu::Typecast<APPROX, in_format, out_format, is_fp32_dest_acc_en>;
    // The pairs the unpacker/packer convert on their own run no SFPU op.
    if constexpr (Op::has_kernel) {
        Op::run(idst);
    }
#endif
}

/**
 * Please refer to documentation for any_init.
 */
template <std::uint32_t IN_DTYPE, std::uint32_t OUT_DTYPE>
ALWI void typecast_tile_init() {
#ifdef TRISC_MATH
    constexpr DataFormat in_format = static_cast<DataFormat>(IN_DTYPE);
    constexpr DataFormat out_format = static_cast<DataFormat>(OUT_DTYPE);
    using Op = sfpu::Typecast<APPROX, in_format, out_format, DST_ACCUM_MODE>;
    // Where the pairs that run no SFPU op need no init either (Quasar), needs_init is false.
    if constexpr (Op::needs_init) {
        Op::init();
    }
#endif
}

}  // namespace ckernel

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_typecast.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
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
 *  Float32 <-> Int32
 *  Float32 <-> UInt16
 *  Float32 <-> UInt32
 *  Bfp8_b <-> Int32
 *  Bfp8_b <-> UInt16
 *  Bfp8_b <-> UInt32
 *  Bfp8_b <-> Float16_b
 *  Bfp8_b <-> Float32
 *  Bfp4_b <-> Int32
 *  Bfp4_b <-> UInt16
 *  Bfp4_b <-> UInt32
 *  Bfp4_b <-> Bfp8_b
 *  Bfp4_b <-> Float16_b
 *  Bfp4_b <-> Float32
 *  UInt16 -> UInt32
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
template <uint32_t IN_DTYPE, uint32_t OUT_DTYPE>
ALWI void typecast_tile(uint32_t idst) {
    MATH(({
        if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float16_b && OUT_DTYPE == (uint32_t)DataFormat::UInt16) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_fp16b_to_uint16<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt16 && OUT_DTYPE == (uint32_t)DataFormat::Float16_b) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_uint16_to_fp16b<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Int32 && OUT_DTYPE == (uint32_t)DataFormat::Float16_b) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_int32_to_fp16b<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float16_b && OUT_DTYPE == (uint32_t)DataFormat::Int32) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_fp16b_to_int32<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (
            IN_DTYPE == (uint32_t)DataFormat::Float16_b && OUT_DTYPE == (uint32_t)DataFormat::Float32) {
            // no SFPU kernel needed, handled by packer
        } else if constexpr (
            IN_DTYPE == (uint32_t)DataFormat::Float32 && OUT_DTYPE == (uint32_t)DataFormat::Float16_b) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_fp32_to_fp16b<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float32 && OUT_DTYPE == (uint32_t)DataFormat::UInt16) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_fp16b_to_uint16<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt16 && OUT_DTYPE == (uint32_t)DataFormat::Float32) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_uint16_to_fp32<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float32 && OUT_DTYPE == (uint32_t)DataFormat::Int32) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_fp16b_to_int32<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Int32 && OUT_DTYPE == (uint32_t)DataFormat::Float32) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_int32_to_fp32<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp8_b && OUT_DTYPE == (uint32_t)DataFormat::UInt16) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_fp16b_to_uint16<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt16 && OUT_DTYPE == (uint32_t)DataFormat::Bfp8_b) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_uint16_to_fp16b<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp8_b && OUT_DTYPE == (uint32_t)DataFormat::Int32) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_fp16b_to_int32<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Int32 && OUT_DTYPE == (uint32_t)DataFormat::Bfp8_b) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_int32_to_fp16b<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float16_b && OUT_DTYPE == (uint32_t)DataFormat::UInt32) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_fp16b_to_uint32<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt32 && OUT_DTYPE == (uint32_t)DataFormat::Float16_b) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_uint32_to_fp16b<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float32 && OUT_DTYPE == (uint32_t)DataFormat::UInt32) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_fp16b_to_uint32<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt32 && OUT_DTYPE == (uint32_t)DataFormat::Float32) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_uint32_to_fp32<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp8_b && OUT_DTYPE == (uint32_t)DataFormat::UInt32) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_fp16b_to_uint32<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt32 && OUT_DTYPE == (uint32_t)DataFormat::Bfp8_b) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_uint32_to_fp16b<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt16 && OUT_DTYPE == (uint32_t)DataFormat::UInt32) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_uint16_to_uint32<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp8_b && OUT_DTYPE == (uint32_t)DataFormat::Float16_b) {
            // no SFPU kernel needed, handled by unpacker
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float16_b && OUT_DTYPE == (uint32_t)DataFormat::Bfp8_b) {
            // no SFPU kernel needed, handled by packer
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp8_b && OUT_DTYPE == (uint32_t)DataFormat::Float32) {
            // no SFPU kernel needed, handled by unpacker/packer
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float32 && OUT_DTYPE == (uint32_t)DataFormat::Bfp8_b) {
            // no SFPU kernel needed, handled by packer
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp4_b && OUT_DTYPE == (uint32_t)DataFormat::UInt16) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_fp16b_to_uint16<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt16 && OUT_DTYPE == (uint32_t)DataFormat::Bfp4_b) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_uint16_to_fp16b<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp4_b && OUT_DTYPE == (uint32_t)DataFormat::Int32) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_fp16b_to_int32<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Int32 && OUT_DTYPE == (uint32_t)DataFormat::Bfp4_b) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_int32_to_fp16b<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp4_b && OUT_DTYPE == (uint32_t)DataFormat::UInt32) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_fp16b_to_uint32<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt32 && OUT_DTYPE == (uint32_t)DataFormat::Bfp4_b) {
            llk_math_eltwise_unary_sfpu_params<APPROX>(
                ckernel::sfpu::calculate_typecast_uint32_to_fp16b<APPROX, 8>, idst, (int)VectorMode::RC);
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp4_b && OUT_DTYPE == (uint32_t)DataFormat::Float16_b) {
            // no SFPU kernel needed, handled by unpacker
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float16_b && OUT_DTYPE == (uint32_t)DataFormat::Bfp4_b) {
            // no SFPU kernel needed, handled by packer
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp4_b && OUT_DTYPE == (uint32_t)DataFormat::Bfp8_b) {
            // no SFPU kernel needed, handled by unpacker
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp8_b && OUT_DTYPE == (uint32_t)DataFormat::Bfp4_b) {
            // no SFPU kernel needed, handled by packer
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp4_b && OUT_DTYPE == (uint32_t)DataFormat::Float32) {
            // no SFPU kernel needed, handled by unpacker/packer
        } else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float32 && OUT_DTYPE == (uint32_t)DataFormat::Bfp4_b) {
            // no SFPU kernel needed, handled by packer
        }
    }));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void typecast_tile_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROX>())); }

}  // namespace ckernel

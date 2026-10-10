// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_quant.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

#ifdef TRISC_MATH
namespace detail {
#ifdef ARCH_BLACKHOLE
// Blackhole quantizes a tile in one 32-row call.
inline constexpr int quant_iterations = 32;
inline constexpr VectorMode quant_vector_mode = VectorMode::None;
#else
inline constexpr int quant_iterations = 8;
inline constexpr VectorMode quant_vector_mode = VectorMode::RC;
#endif
}  // namespace detail
#endif

// clang-format off
/**
 * Performs an elementwise per-tensor affine quantization operation on the first operand using the scaling factor in the second operand.
 * Output overwrites odst in DST.
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
ALWI void quant_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_quant_int32,
        (APPROX, detail::quant_iterations),
        idst0,
        idst1,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Quantize variant writing an int8 output tensor.
 * Output overwrites odst in DST.
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
ALWI void quant_int8_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_quant_int32_int8_pack,
        (APPROX, detail::quant_iterations),
        idst0,
        idst1,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Performs an elementwise per-tensor affine re-quantization operation on the first operand using the scaling factor in the second operand.
 * Output overwrites odst in DST.
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
ALWI void requant_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_requant_int32,
        (APPROX, detail::quant_iterations),
        idst0,
        idst1,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Re-quantize variant writing an int8 output tensor.
 * Output overwrites odst in DST.
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
ALWI void requant_int8_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_requant_int32_int8_pack,
        (APPROX, detail::quant_iterations),
        idst0,
        idst1,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Re-quantize variant reading an int8 input tensor and writing an int32 output.
 * Output overwrites odst in DST.
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
ALWI void requant_int8_in_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_requant_int32,
        (APPROX, detail::quant_iterations, false, true),
        idst0,
        idst1,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Re-quantize variant reading an int8 input tensor and writing an int8 output tensor.
 * int8-input unbias (see requant_int8_in_tile_init) with int8-output packing into [-128, 127].
 * Output overwrites odst in DST.
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
ALWI void requant_int8_in_int8_out_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_requant_int32_int8_pack,
        (APPROX, detail::quant_iterations, true),
        idst0,
        idst1,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Performs an elementwise per-tensor affine de-quantization operation on the first operand using the scaling factor in the second operand.
 * Output overwrites odst in DST.
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
ALWI void dequant_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_dequant_int32,
        (APPROX, detail::quant_iterations),
        idst0,
        idst1,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * De-quantize variant reading an int8 input tensor.
 * Output overwrites odst in DST.
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
ALWI void dequant_int8_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_dequant_int32,
        (APPROX, detail::quant_iterations, false, true),
        idst0,
        idst1,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Initialize the sfpu with the zero point argument of the quantization Op.
 * To be called once at beginning of a kernel.
 *
 * Return value: None
 *
 * | Argument   | Description                           | Data type | Valid range | Required |
 * |------------|---------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void quant_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(quant_int32, sfpu::quant_init, (APPROX), zero_point)));
}

// clang-format off
/**
 * Initialize the sfpu with the zero point argument of the quantization Op, rounding into the
 * unsigned uint8 range [0, 255]. To be called once at beginning of a kernel.
 *
 * Return value: None
 *
 * | Argument   | Description                           | Data type | Valid range | Required |
 * |------------|---------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void quant_uint8_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(quant_int32, sfpu::quant_init, (APPROX, false, DataFormat::UInt8), zero_point)));
}

// clang-format off
/**
 * Initialize the sfpu for quantize with int8 output.
 *
 * Return value: None
 *
 * | Argument   | Description                           | Data type | Valid range | Required |
 * |------------|---------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void quant_int8_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(quant_int32, sfpu::quant_init, (APPROX, false, DataFormat::Int8), zero_point)));
}

// clang-format off
/**
 * Initialize the sfpu with the zero point argument of the re-quantization Op.
 * To be called once at beginning of a kernel.
 *
 * Return value: None
 *
 * | Argument   | Description                              | Data type | Valid range | Required |
 * |------------|------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the re-quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void requant_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(requant_int32, sfpu::requant_init, (APPROX), zero_point)));
}

// clang-format off
/**
 * Initialize the sfpu with the zero point argument of the re-quantization Op, rounding into the
 * unsigned uint8 range [0, 255]. To be called once at beginning of a kernel.
 *
 * Return value: None
 *
 * | Argument   | Description                              | Data type | Valid range | Required |
 * |------------|------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the re-quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void requant_uint8_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(requant_int32, sfpu::requant_init, (APPROX, false, DataFormat::UInt8), zero_point)));
}

// clang-format off
/**
 * Initialize the sfpu for requantize with int8 output.
 *
 * Return value: None
 *
 * | Argument   | Description                              | Data type | Valid range | Required |
 * |------------|------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the re-quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void requant_int8_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(requant_int32, sfpu::requant_init, (APPROX, false, DataFormat::Int8), zero_point)));
}

// clang-format off
/**
 * Initialize the sfpu for requantize reading an int8 input tensor. Must be called before using the requant
 * int8-input tile op. Some binary_ng kernels invoke this init inside the per-tile loop. Repeated calls are redundant.
 *
 * Return value: None
 *
 * | Argument   | Description                              | Data type | Valid range | Required |
 * |------------|------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the re-quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void requant_int8_in_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(
        requant_int32, sfpu::requant_init, (APPROX, false, DataFormat::Int32, true), zero_point)));
}

// clang-format off
/**
 * Initialize the sfpu for requantize reading an int8 input tensor and writing a uint8 output tensor. Shares the
 * int8-input handling of requant_int8_in_tile_init (see that function). Must be called before using the op.
 * Repeated calls are redundant.
 *
 * Return value: None
 *
 * | Argument   | Description                              | Data type | Valid range | Required |
 * |------------|------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the re-quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void requant_int8_in_uint8_out_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(
        requant_int32, sfpu::requant_init, (APPROX, false, DataFormat::UInt8, true), zero_point)));
}

// clang-format off
/**
 * Initialize the sfpu for requantize reading an int8 input tensor and writing an int8 output tensor. Shares the
 * int8-input handling of requant_int8_in_tile_init and additionally packs the result into the signed int8 range.
 * Must be called before using the op. Repeated calls are redundant.
 *
 * Return value: None
 *
 * | Argument   | Description                              | Data type | Valid range | Required |
 * |------------|------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the re-quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void requant_int8_in_int8_out_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(
        requant_int32, sfpu::requant_init, (APPROX, false, DataFormat::Int8, true), zero_point)));
}

// clang-format off
/**
 * Initialize the sfpu with the zero point argument of the de-quantization Op.
 * To be called once at beginning of a kernel.
 *
 * Return value: None
 *
 * | Argument   | Description                              | Data type | Valid range | Required |
 * |------------|------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the de-quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void dequant_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(dequant_int32, sfpu::dequant_init, (APPROX), zero_point)));
}

// clang-format off
/**
 * Initialize the sfpu for dequantize reading an int8 input tensor. Must be called before using the dequant
 * int8-input tile op. Some binary_ng kernels invoke this init inside the per-tile loop. Repeated calls are redundant.
 *
 * Return value: None
 *
 * | Argument   | Description                              | Data type | Valid range | Required |
 * |------------|------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the de-quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void dequant_int8_tile_init(const uint32_t zero_point) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(dequant_int32, sfpu::dequant_init, (APPROX, false, true), zero_point)));
}

#if defined(ARCH_BLACKHOLE)
// clang-format off
/**
 * Blackhole only. Per-tensor form of quant_tile: the scale comes from quant_scalar_tile_init or
 * quant_uint8_scalar_tile_init instead of a second DEST tile, so the call has one operand. Dest must be in 32 bit mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to use as the operand    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void quant_scalar_tile(uint32_t idst, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_quant_int32,
        (APPROX, detail::quant_iterations, false, true /*SCALAR_SCALE*/),
        idst,
        idst /*dst_index_in1, unused*/,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Blackhole only. Per-tensor form of quant_int8_tile: the scale comes from quant_int8_scalar_tile_init instead of a
 * second DEST tile, so the call has one operand. Dest must be in 32 bit mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to use as the operand    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void quant_int8_scalar_tile(uint32_t idst, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_quant_int32_int8_pack,
        (APPROX, detail::quant_iterations, true /*SCALAR_SCALE*/),
        idst,
        idst /*dst_index_in1, unused*/,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Blackhole only. Per-tensor form of requant_tile: the scale comes from requant_scalar_tile_init or
 * requant_uint8_scalar_tile_init instead of a second DEST tile, so the call has one operand. Dest must be in 32 bit
 * mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to use as the operand    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void requant_scalar_tile(uint32_t idst, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_requant_int32,
        (APPROX, detail::quant_iterations, false, false, true /*SCALAR_SCALE*/),
        idst,
        idst /*dst_index_in1, unused*/,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Blackhole only. Per-tensor form of requant_int8_tile: the scale comes from requant_int8_scalar_tile_init instead of a
 * second DEST tile, so the call has one operand. Dest must be in 32 bit mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to use as the operand    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void requant_int8_scalar_tile(uint32_t idst, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_requant_int32_int8_pack,
        (APPROX, detail::quant_iterations, false, true /*SCALAR_SCALE*/),
        idst,
        idst /*dst_index_in1, unused*/,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Blackhole only. Per-tensor form of requant_int8_in_tile: the scale comes from requant_int8_in_scalar_tile_init or
 * requant_int8_in_uint8_out_scalar_tile_init instead of a second DEST tile, so the call has one operand. Dest must be
 * in 32 bit mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to use as the operand    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void requant_int8_in_scalar_tile(uint32_t idst, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_requant_int32,
        (APPROX, detail::quant_iterations, false, true, true /*SCALAR_SCALE*/),
        idst,
        idst /*dst_index_in1, unused*/,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Blackhole only. Per-tensor form of requant_int8_in_int8_out_tile: the scale comes from
 * requant_int8_in_int8_out_scalar_tile_init instead of a second DEST tile, so the call has one operand. Dest must be in
 * 32 bit mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to use as the operand    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void requant_int8_in_int8_out_scalar_tile(uint32_t idst, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_requant_int32_int8_pack,
        (APPROX, detail::quant_iterations, true, true /*SCALAR_SCALE*/),
        idst,
        idst /*dst_index_in1, unused*/,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Blackhole only. Per-tensor form of dequant_tile: the scale comes from dequant_scalar_tile_init instead of a second
 * DEST tile, so the call has one operand. Dest must be in 32 bit mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to use as the operand    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void dequant_scalar_tile(uint32_t idst, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_dequant_int32,
        (APPROX, detail::quant_iterations, false, false, true /*SCALAR_SCALE*/),
        idst,
        idst /*dst_index_in1, unused*/,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Blackhole only. Per-tensor form of dequant_int8_tile: the scale comes from dequant_int8_scalar_tile_init instead of a
 * second DEST tile, so the call has one operand. Dest must be in 32 bit mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to use as the operand    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void dequant_int8_scalar_tile(uint32_t idst, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_dequant_int32,
        (APPROX, detail::quant_iterations, false, true, true /*SCALAR_SCALE*/),
        idst,
        idst /*dst_index_in1, unused*/,
        odst,
        detail::quant_vector_mode)));
}

// clang-format off
/**
 * Blackhole only. Initializes the SFPU for quant_scalar_tile with the zero point and the per-tensor scale of the Op.
 * The scale stays in an SFPU register, so no other SFPU op may run between this init and the quant_scalar_tile calls.
 *
 * Return value: None
 *
 * | Argument   | Description                                   | Data type | Valid range | Required |
 * |------------|-----------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the Op (fp32 bits)          | uint32_t  | Any number  | Yes      |
 * | scale      | The per-tensor scale of the Op (fp32 bits)    | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void quant_scalar_tile_init(const uint32_t zero_point, const uint32_t scale) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(quant_int32, sfpu::quant_init_scalar_scale, (APPROX), zero_point, scale)));
}

// clang-format off
/**
 * Blackhole only. Initializes the SFPU for quant_scalar_tile with the zero point and the per-tensor scale of the Op,
 * rounding into the unsigned uint8 range [0, 255]. The scale stays in an SFPU register, so no other SFPU op may run
 * between this init and the quant_scalar_tile calls.
 *
 * Return value: None
 *
 * | Argument   | Description                                   | Data type | Valid range | Required |
 * |------------|-----------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the Op (fp32 bits)          | uint32_t  | Any number  | Yes      |
 * | scale      | The per-tensor scale of the Op (fp32 bits)    | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void quant_uint8_scalar_tile_init(const uint32_t zero_point, const uint32_t scale) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(
        quant_int32, sfpu::quant_init_scalar_scale, (APPROX, false, DataFormat::UInt8), zero_point, scale)));
}

// clang-format off
/**
 * Blackhole only. Initializes the SFPU for quant_int8_scalar_tile with the zero point and the per-tensor scale of the
 * Op. The scale stays in an SFPU register, so no other SFPU op may run between this init and the quant_int8_scalar_tile
 * calls.
 *
 * Return value: None
 *
 * | Argument   | Description                                   | Data type | Valid range | Required |
 * |------------|-----------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the Op (fp32 bits)          | uint32_t  | Any number  | Yes      |
 * | scale      | The per-tensor scale of the Op (fp32 bits)    | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void quant_int8_scalar_tile_init(const uint32_t zero_point, const uint32_t scale) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(
        quant_int32, sfpu::quant_init_scalar_scale, (APPROX, false, DataFormat::Int8), zero_point, scale)));
}

// clang-format off
/**
 * Blackhole only. Initializes the SFPU for requant_scalar_tile with the zero point and the per-tensor scale of the Op.
 * The scale stays in an SFPU register, so no other SFPU op may run between this init and the requant_scalar_tile calls.
 *
 * Return value: None
 *
 * | Argument   | Description                                   | Data type | Valid range | Required |
 * |------------|-----------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the Op (fp32 bits)          | uint32_t  | Any number  | Yes      |
 * | scale      | The per-tensor scale of the Op (fp32 bits)    | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void requant_scalar_tile_init(const uint32_t zero_point, const uint32_t scale) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(requant_int32, sfpu::requant_init_scalar_scale, (APPROX), zero_point, scale)));
}

// clang-format off
/**
 * Blackhole only. Initializes the SFPU for requant_scalar_tile with the zero point and the per-tensor scale of the Op,
 * rounding into the unsigned uint8 range [0, 255]. The scale stays in an SFPU register, so no other SFPU op may run
 * between this init and the requant_scalar_tile calls.
 *
 * Return value: None
 *
 * | Argument   | Description                                   | Data type | Valid range | Required |
 * |------------|-----------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the Op (fp32 bits)          | uint32_t  | Any number  | Yes      |
 * | scale      | The per-tensor scale of the Op (fp32 bits)    | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void requant_uint8_scalar_tile_init(const uint32_t zero_point, const uint32_t scale) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(
        requant_int32, sfpu::requant_init_scalar_scale, (APPROX, false, DataFormat::UInt8), zero_point, scale)));
}

// clang-format off
/**
 * Blackhole only. Initializes the SFPU for requant_int8_scalar_tile with the zero point and the per-tensor scale of the
 * Op. The scale stays in an SFPU register, so no other SFPU op may run between this init and the
 * requant_int8_scalar_tile calls.
 *
 * Return value: None
 *
 * | Argument   | Description                                   | Data type | Valid range | Required |
 * |------------|-----------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the Op (fp32 bits)          | uint32_t  | Any number  | Yes      |
 * | scale      | The per-tensor scale of the Op (fp32 bits)    | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void requant_int8_scalar_tile_init(const uint32_t zero_point, const uint32_t scale) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(
        requant_int32, sfpu::requant_init_scalar_scale, (APPROX, false, DataFormat::Int8), zero_point, scale)));
}

// clang-format off
/**
 * Blackhole only. Initializes the SFPU for requant_int8_in_scalar_tile with the zero point and the per-tensor scale of
 * the Op. The scale stays in an SFPU register, so no other SFPU op may run between this init and the
 * requant_int8_in_scalar_tile calls.
 *
 * Return value: None
 *
 * | Argument   | Description                                   | Data type | Valid range | Required |
 * |------------|-----------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the Op (fp32 bits)          | uint32_t  | Any number  | Yes      |
 * | scale      | The per-tensor scale of the Op (fp32 bits)    | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void requant_int8_in_scalar_tile_init(const uint32_t zero_point, const uint32_t scale) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(
        requant_int32, sfpu::requant_init_scalar_scale, (APPROX, false, DataFormat::Int32, true), zero_point, scale)));
}

// clang-format off
/**
 * Blackhole only. Initializes the SFPU for requant_int8_in_scalar_tile with the zero point and the per-tensor scale of
 * the Op, rounding into the unsigned uint8 range [0, 255]. The scale stays in an SFPU register, so no other SFPU op may
 * run between this init and the requant_int8_in_scalar_tile calls.
 *
 * Return value: None
 *
 * | Argument   | Description                                   | Data type | Valid range | Required |
 * |------------|-----------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the Op (fp32 bits)          | uint32_t  | Any number  | Yes      |
 * | scale      | The per-tensor scale of the Op (fp32 bits)    | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void requant_int8_in_uint8_out_scalar_tile_init(const uint32_t zero_point, const uint32_t scale) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(
        requant_int32, sfpu::requant_init_scalar_scale, (APPROX, false, DataFormat::UInt8, true), zero_point, scale)));
}

// clang-format off
/**
 * Blackhole only. Initializes the SFPU for requant_int8_in_int8_out_scalar_tile with the zero point and the per-tensor
 * scale of the Op. The scale stays in an SFPU register, so no other SFPU op may run between this init and the
 * requant_int8_in_int8_out_scalar_tile calls.
 *
 * Return value: None
 *
 * | Argument   | Description                                   | Data type | Valid range | Required |
 * |------------|-----------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the Op (fp32 bits)          | uint32_t  | Any number  | Yes      |
 * | scale      | The per-tensor scale of the Op (fp32 bits)    | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void requant_int8_in_int8_out_scalar_tile_init(const uint32_t zero_point, const uint32_t scale) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(
        requant_int32, sfpu::requant_init_scalar_scale, (APPROX, false, DataFormat::Int8, true), zero_point, scale)));
}

// clang-format off
/**
 * Blackhole only. Initializes the SFPU for dequant_scalar_tile with the zero point and the per-tensor scale of the Op.
 * The scale stays in an SFPU register, so no other SFPU op may run between this init and the dequant_scalar_tile calls.
 *
 * Return value: None
 *
 * | Argument   | Description                                   | Data type | Valid range | Required |
 * |------------|-----------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the Op (fp32 bits)          | uint32_t  | Any number  | Yes      |
 * | scale      | The per-tensor scale of the Op (fp32 bits)    | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void dequant_scalar_tile_init(const uint32_t zero_point, const uint32_t scale) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(dequant_int32, sfpu::dequant_init_scalar_scale, (APPROX), zero_point, scale)));
}

// clang-format off
/**
 * Blackhole only. Initializes the SFPU for dequant_int8_scalar_tile with the zero point and the per-tensor scale of the
 * Op. The scale stays in an SFPU register, so no other SFPU op may run between this init and the
 * dequant_int8_scalar_tile calls.
 *
 * Return value: None
 *
 * | Argument   | Description                                   | Data type | Valid range | Required |
 * |------------|-----------------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the Op (fp32 bits)          | uint32_t  | Any number  | Yes      |
 * | scale      | The per-tensor scale of the Op (fp32 bits)    | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void dequant_int8_scalar_tile_init(const uint32_t zero_point, const uint32_t scale) {
    MATH((SFPU_BINARY_INIT_FN_ARGS(
        dequant_int32, sfpu::dequant_init_scalar_scale, (APPROX, false, true), zero_point, scale)));
}
#endif  // ARCH_BLACKHOLE

}  // namespace ckernel

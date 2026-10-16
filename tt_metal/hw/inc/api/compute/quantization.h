// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_quant.h"
#endif

namespace ckernel {

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
ALWI void quant_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::Quant<APPROX>::run(idst0, idst1, odst)));
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
ALWI void quant_int8_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::Quant<APPROX, DataFormat::Int8>::run(idst0, idst1, odst)));
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
ALWI void requant_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::Requant<APPROX>::run(idst0, idst1, odst)));
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
ALWI void requant_int8_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::Requant<APPROX, DataFormat::Int8>::run(idst0, idst1, odst)));
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
ALWI void requant_int8_in_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::Requant<APPROX, DataFormat::Int32, true /* INT8_INPUT */>::run(idst0, idst1, odst)));
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
ALWI void requant_int8_in_int8_out_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::Requant<APPROX, DataFormat::Int8, true /* INT8_INPUT */>::run(idst0, idst1, odst)));
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
ALWI void dequant_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::Dequant<APPROX>::run(idst0, idst1, odst)));
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
ALWI void dequant_int8_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::Dequant<APPROX, true /* INT8_INPUT */>::run(idst0, idst1, odst)));
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
ALWI void quant_tile_init(const std::uint32_t zero_point) { MATH((sfpu::Quant<APPROX>::init(zero_point))); }

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
ALWI void quant_uint8_tile_init(const std::uint32_t zero_point) {
    MATH((sfpu::Quant<APPROX, DataFormat::UInt8>::init(zero_point)));
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
ALWI void quant_int8_tile_init(const std::uint32_t zero_point) {
    MATH((sfpu::Quant<APPROX, DataFormat::Int8>::init(zero_point)));
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
ALWI void requant_tile_init(const std::uint32_t zero_point) { MATH((sfpu::Requant<APPROX>::init(zero_point))); }

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
ALWI void requant_uint8_tile_init(const std::uint32_t zero_point) {
    MATH((sfpu::Requant<APPROX, DataFormat::UInt8>::init(zero_point)));
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
ALWI void requant_int8_tile_init(const std::uint32_t zero_point) {
    MATH((sfpu::Requant<APPROX, DataFormat::Int8>::init(zero_point)));
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
ALWI void requant_int8_in_tile_init(const std::uint32_t zero_point) {
    MATH((sfpu::Requant<APPROX, DataFormat::Int32, true /* INT8_INPUT */>::init(zero_point)));
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
ALWI void requant_int8_in_uint8_out_tile_init(const std::uint32_t zero_point) {
    MATH((sfpu::Requant<APPROX, DataFormat::UInt8, true /* INT8_INPUT */>::init(zero_point)));
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
ALWI void requant_int8_in_int8_out_tile_init(const std::uint32_t zero_point) {
    MATH((sfpu::Requant<APPROX, DataFormat::Int8, true /* INT8_INPUT */>::init(zero_point)));
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
ALWI void dequant_tile_init(const std::uint32_t zero_point) { MATH((sfpu::Dequant<APPROX>::init(zero_point))); }

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
ALWI void dequant_int8_tile_init(const std::uint32_t zero_point) {
    MATH((sfpu::Dequant<APPROX, true /* INT8_INPUT */>::init(zero_point)));
}

}  // namespace ckernel

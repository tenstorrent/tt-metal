// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "tensor_shape.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary_max_min.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise maximum operation on inputs of int32 data type at idst0, idst1: y = max(x0, x1).
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
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
ALWI void binary_max_int32_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::BinaryMaxMinInt32<true /* IS_MAX */, false /* IS_UNSIGNED */>::run(idst0, idst1, odst)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_max_int32_tile_init() {
    MATH((sfpu::BinaryMaxMinInt32<true /* IS_MAX */, false /* IS_UNSIGNED */>::init()));
}

// clang-format off
/**
 * Performs an elementwise maximum operation on inputs of uint32 data type at idst0, idst1: y = max(x0, x1).
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
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
// Quasar has no uint32 max/min.
#ifndef ARCH_QUASAR
ALWI void binary_max_uint32_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::BinaryMaxMinInt32<true /* IS_MAX */, true /* IS_UNSIGNED */>::run(idst0, idst1, odst)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_max_uint32_tile_init() {
    MATH((sfpu::BinaryMaxMinInt32<true /* IS_MAX */, true /* IS_UNSIGNED */>::init()));
}
#endif

// clang-format off
/**
 * Performs an elementwise maximum operation on inputs at idst0, idst1: y = max(x0, x1).
 * Output overwrites odst in DST.
 *
 * The TENSOR_SHAPE template parameter selects the tile to process, e.g.
 * tensor_shape_from_tile_dims(32, 16) for the left column of faces; the default is the full 32x32 tile.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
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
template <TensorShape TENSOR_SHAPE = DEFAULT_TENSOR_SHAPE>
ALWI void binary_max_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::BinaryMaxMin<true /* IS_MAX */>::run<TENSOR_SHAPE>(idst0, idst1, odst)));
}

/**
 * Legacy overload selecting the faces to process with a VectorMode. Prefer the TensorShape template
 * parameter of the overload above: VectorMode::R and VectorMode::C correspond to
 * tensor_shape_from_tile_dims(16, 32) and tensor_shape_from_tile_dims(32, 16).
 */
ALWI void binary_max_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode) {
    MATH((sfpu::BinaryMaxMin<true /* IS_MAX */>::run_vector_mode(vector_mode, idst0, idst1, odst)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_max_tile_init() { MATH((sfpu::BinaryMaxMin<true /* IS_MAX */>::init())); }

// clang-format off
/**
 * Performs an elementwise minimum operation on inputs of int32 data type at idst0, idst1: y = min(x0, x1).
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
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
ALWI void binary_min_int32_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::BinaryMaxMinInt32<false /* IS_MAX */, false /* IS_UNSIGNED */>::run(idst0, idst1, odst)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_min_int32_tile_init() {
    MATH((sfpu::BinaryMaxMinInt32<false /* IS_MAX */, false /* IS_UNSIGNED */>::init()));
}

// clang-format off
/**
 * Performs an elementwise minimum operation on inputs of uint32 data type at idst0, idst1: y = min(x0, x1).
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
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
// Quasar has no uint32 max/min.
#ifndef ARCH_QUASAR
ALWI void binary_min_uint32_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::BinaryMaxMinInt32<false /* IS_MAX */, true /* IS_UNSIGNED */>::run(idst0, idst1, odst)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_min_uint32_tile_init() {
    MATH((sfpu::BinaryMaxMinInt32<false /* IS_MAX */, true /* IS_UNSIGNED */>::init()));
}
#endif

// clang-format off
/**
 * Performs an elementwise minimum operation on inputs at idst0, idst1: y = min(x0, x1).
 * Output overwrites odst in DST.
 *
 * The TENSOR_SHAPE template parameter selects the tile to process, e.g.
 * tensor_shape_from_tile_dims(32, 16) for the left column of faces; the default is the full 32x32 tile.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
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
template <TensorShape TENSOR_SHAPE = DEFAULT_TENSOR_SHAPE>
ALWI void binary_min_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::BinaryMaxMin<false /* IS_MAX */>::run<TENSOR_SHAPE>(idst0, idst1, odst)));
}

/**
 * Legacy overload selecting the faces to process with a VectorMode. Prefer the TensorShape template
 * parameter of the overload above: VectorMode::R and VectorMode::C correspond to
 * tensor_shape_from_tile_dims(16, 32) and tensor_shape_from_tile_dims(32, 16).
 */
ALWI void binary_min_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode) {
    MATH((sfpu::BinaryMaxMin<false /* IS_MAX */>::run_vector_mode(vector_mode, idst0, idst1, odst)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_min_tile_init() { MATH((sfpu::BinaryMaxMin<false /* IS_MAX */>::init())); }

}  // namespace ckernel

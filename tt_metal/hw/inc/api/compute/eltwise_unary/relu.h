// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_relu.h"
#endif

namespace ckernel {

ALWI void relu_tile_init() { MATH((sfpu::ReluMin<sfpi::vFloat, APPROX>::init())); }

// clang-format off
/**
 * Performs element-wise computation of relu(x) = (0 if x is negative else x) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void relu_tile(std::uint32_t idst) { MATH((sfpu::ReluMin<sfpi::vFloat, APPROX>::run(idst, 0 /* threshold */))); }

// clang-format off
/**
 * Performs element-wise computation of relu min (relu(min(x, lower_limit))) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | lower_limit    | Upper limit of relu_min                                                    | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on
ALWI void relu_min_tile(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::ReluMin<sfpi::vFloat, APPROX>::run(idst, param0 /* threshold */)));
}

ALWI void relu_min_tile_init() { MATH((sfpu::ReluMin<sfpi::vFloat, APPROX>::init())); }

// clang-format off
/**
 * Performs element-wise computation of relu max (relu(max(x, upper_limit))) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | upper_limit    | Upper limit of relu_min                                                    | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on
ALWI void relu_max_tile(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::ReluMax<sfpi::vFloat, APPROX>::run(idst, param0 /* threshold */)));
}

ALWI void relu_max_tile_init() { MATH((sfpu::ReluMax<sfpi::vFloat, APPROX>::init())); }

// Quasar has no pack-thread SFPU and no integer relu kernels.
#ifndef ARCH_QUASAR
ALWI void relu_max_tile_pack(std::uint32_t idst, std::uint32_t param0) {
    PACK((sfpu::ReluMax<sfpi::vFloat, APPROX>::run(idst, param0 /* threshold */)));
}

ALWI void relu_max_tile_int32(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::ReluClampInt<APPROX, false /* IS_LOWER_BOUND */>::run(idst, param0 /* threshold */)));
}

ALWI void relu_max_tile_uint32(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::ReluClampUint<APPROX, false /* IS_LOWER_BOUND */, DataFormat::UInt32>::run(
        idst, param0 /* threshold */)));
}

ALWI void relu_max_tile_uint16(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::ReluClampUint<APPROX, false /* IS_LOWER_BOUND */, DataFormat::UInt16>::run(
        idst, param0 /* threshold */)));
}

ALWI void relu_max_tile_init_pack() { PACK((sfpu::ReluMax<sfpi::vFloat, APPROX>::init())); }

ALWI void relu_min_tile_int32(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::ReluClampInt<APPROX, true /* IS_LOWER_BOUND */>::run(idst, param0 /* threshold */)));
}

ALWI void relu_min_tile_uint32(std::uint32_t idst, std::uint32_t param0) {
    MATH((
        sfpu::ReluClampUint<APPROX, true /* IS_LOWER_BOUND */, DataFormat::UInt32>::run(idst, param0 /* threshold */)));
}

ALWI void relu_min_tile_uint16(std::uint32_t idst, std::uint32_t param0) {
    MATH((
        sfpu::ReluClampUint<APPROX, true /* IS_LOWER_BOUND */, DataFormat::UInt16>::run(idst, param0 /* threshold */)));
}

ALWI void relu_tile_int32(std::uint32_t idst) {
    MATH((sfpu::ReluMin<sfpi::vInt, APPROX>::run(idst, 0 /* threshold */)));
}

#endif  // !ARCH_QUASAR

// clang-format off
/**
 * Performs element-wise computation of leaky relu (relu(x) + slope*-relu(-x)) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | slope          | slope used in leaky relu - will reinterpret unsigned int to float          | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on
ALWI void leaky_relu_tile(std::uint32_t idst, std::uint32_t slope = 0) {
    MATH((sfpu::Lrelu<APPROX>::run(idst, slope)));
}

ALWI void leaky_relu_tile_init() { MATH((sfpu::Lrelu<APPROX>::init())); }
}  // namespace ckernel

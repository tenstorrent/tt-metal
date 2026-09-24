// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_rounding_ops.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
// The rounding ops need no init beyond the op-agnostic SFPU init.
ALWI void rounding_op_tile_init() { MATH((sfpu::UnaryFn::init())); }

// clang-format off
/**
 * Performs element-wise ceil computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform ceil operation     | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void ceil_tile(std::uint32_t idst) { MATH((sfpu::Ceil<APPROX>::run(idst))); }

// clang-format off
/**
 * Performs element-wise floor computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform floor operation    | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void floor_tile(std::uint32_t idst) { MATH((sfpu::Floor<APPROX>::run(idst))); }

// clang-format off
/**
 * Performs element-wise trunc computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform trunc operation    | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void trunc_tile(std::uint32_t idst) { MATH((sfpu::Trunc<APPROX>::run(idst))); }

// clang-format off
/**
 * Performs element-wise computation of the round operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | decimals        | The number of decimal places to round to.                                  | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void round_tile(std::uint32_t idst, std::int32_t decimals) { MATH((sfpu::Round<APPROX>::run(idst, decimals))); }

// clang-format off
/**
 * Performs element-wise stochastic rounding operation of FP32 values to BF16
 * on each element of a tile in DST register at index tile_index.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Note: this operation uses the SFPSTOCHRND instruction; see the known issues on Wormhole and Blackhole:
 * https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatFloat.md
 * https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatFloat.md
 *
 * Return value: None
 *
 * | Argument        | Description                                                                         | Type     | Valid Range                                           | Required |
 * |-----------------|-------------------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the stochastic round on     | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void stochastic_round_tile(std::uint32_t idst) { MATH((sfpu::StochasticRound<APPROX>::run(idst))); }

// clang-format off
/**
 * Performs element-wise frac computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform frac operation     | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void frac_tile(std::uint32_t idst) { MATH((sfpu::Frac<APPROX>::run(idst))); }

}  // namespace ckernel

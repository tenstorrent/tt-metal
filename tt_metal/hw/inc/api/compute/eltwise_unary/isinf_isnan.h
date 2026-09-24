// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_isinf_isnan.h"
#endif

namespace ckernel {
// clang-format off
/**
 * Will store in the output of the compute core True if the input tile is infinity.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void isinf_tile(std::uint32_t idst) { MATH((sfpu::IsinfIsnan<APPROX, sfpu::FiniteCheck::isinf>::run(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void isinf_tile_init() { MATH((sfpu::IsinfIsnan<APPROX, sfpu::FiniteCheck::isinf>::init())); }

// clang-format off
/**
 * Will store in the output of the compute core True if the input tile is positive infinity.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void isposinf_tile(std::uint32_t idst) {
    MATH((sfpu::IsinfIsnan<APPROX, sfpu::FiniteCheck::isposinf>::run(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isposinf_tile_init() { MATH((sfpu::IsinfIsnan<APPROX, sfpu::FiniteCheck::isposinf>::init())); }

// clang-format off
/**
 * Will store in the output of the compute core True if the input tile is negative infinity.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void isneginf_tile(std::uint32_t idst) {
    MATH((sfpu::IsinfIsnan<APPROX, sfpu::FiniteCheck::isneginf>::run(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isneginf_tile_init() { MATH((sfpu::IsinfIsnan<APPROX, sfpu::FiniteCheck::isneginf>::init())); }

// clang-format off
/**
 * Will store in the output of the compute core True if the input tile is nan.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void isnan_tile(std::uint32_t idst) { MATH((sfpu::IsinfIsnan<APPROX, sfpu::FiniteCheck::isnan>::run(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void isnan_tile_init() { MATH((sfpu::IsinfIsnan<APPROX, sfpu::FiniteCheck::isnan>::init())); }

// clang-format off
/**
 * Will store in the output of the compute core True if the input tile is finite
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void isfinite_tile(std::uint32_t idst) {
    MATH((sfpu::IsinfIsnan<APPROX, sfpu::FiniteCheck::isfinite>::run(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isfinite_tile_init() { MATH((sfpu::IsinfIsnan<APPROX, sfpu::FiniteCheck::isfinite>::init())); }
}  // namespace ckernel

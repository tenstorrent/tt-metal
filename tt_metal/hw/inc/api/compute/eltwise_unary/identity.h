// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_identity.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs a simple elementwise copy / identity operation on the input: y(x) = x
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform identity operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void identity_tile(std::uint32_t idst) { MATH((sfpu::Identity<APPROX>::run(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void identity_tile_init() { MATH((sfpu::Identity<APPROX>::init())); }

// clang-format off
/**
 * Performs a simple elementwise copy / identity operation on the input: y(x) = x
 * This function should be used with unsigned integer formats: uint32
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform identity operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void identity_tile_uint32(std::uint32_t idst) { MATH((sfpu::IdentityUint<APPROX>::run(idst))); }

}  // namespace ckernel

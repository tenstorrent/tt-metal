// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_fmod.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise fmod computation on input x by y , where x is each element of a tile
 * in DST register at index tile_index. The input can be of float data type. The denominator is provided to
 * fmod_tile_init and loaded into the SFPU constant registers. The
 * DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on
 * the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                 | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform fmod operation      | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void fmod_tile(std::uint32_t idst) {
    // The denominator and its reciprocal are loaded once by fmod_tile_init into vConstFloatPrgm0/1;
    // calculate_fmod reads them from there.
    MATH((sfpu::Fmod<APPROX>::run(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void fmod_tile_init(std::uint32_t param0, std::uint32_t param1) {
    MATH((sfpu::Fmod<APPROX>::init(param0, param1)));
}

}  // namespace ckernel

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#ifndef ARCH_QUASAR
#include "ckernel_sfpu_erf.h"
#include "ckernel_sfpu_erfc.h"
#endif
#endif

namespace ckernel {
#ifndef ARCH_QUASAR
/************** ERF *****************/
/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = true>
ALWI void erf_tile_init() {
    MATH((sfpu::Erf<fast_and_approx>::init()));
}

// clang-format off
/**
 * Performs element-wise computation of error function on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool fast_and_approx = true>
ALWI void erf_tile(std::uint32_t idst) {
    MATH((sfpu::Erf<fast_and_approx>::run(idst)));
}

/************** ERFC *****************/

/**
 * Please refer to documentation for any_init.
 */
ALWI void erfc_tile_init() { MATH((sfpu::Erfc<true /* APPROXIMATION_MODE */>::init())); }

// clang-format off
/**
 * Performs element-wise computation of complimentary error function on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void erfc_tile(std::uint32_t idst) { MATH((sfpu::Erfc<true /* APPROXIMATION_MODE */>::run(idst))); }

#endif

}  // namespace ckernel

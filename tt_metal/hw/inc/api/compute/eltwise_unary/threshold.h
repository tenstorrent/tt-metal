// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_threshold.h"
#endif

namespace ckernel {

// clang-format off
/**
* Performs element-wise threshold operation. The DST
* register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
* compute engine.
*
* Return value: None
*
* | Argument        | Description                                                                         | Type     | Valid Range                                           | Required |
* |-----------------|-------------------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on          | uint32_t | Must be less than the size of the DST register buffer | True     |
* | param0          | The threshold value for the threshold function                                      | uint32_t |                                                       | True     |
* | param1          | The value to replace the input with if it is less than or equal to the threshold    | uint32_t |                                                       | True     |
*/
// clang-format on
ALWI void threshold_tile(std::uint32_t idst, std::uint32_t param0, std::uint32_t param1) {
    MATH((sfpu::Threshold<APPROX>::run(idst, param0, param1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void threshold_tile_init() { MATH((sfpu::Threshold<APPROX>::init())); }

}  // namespace ckernel

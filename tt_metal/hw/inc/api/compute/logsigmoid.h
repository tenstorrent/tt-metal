// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_logsigmoid.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise logsigmoid operation on the input: y = log(1 / (1 + exp(-x))).
 * Output overwrites the input in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the operation on   | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void logsigmoid_tile(std::uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_logsigmoid, (APPROX, DST_ACCUM_MODE), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void logsigmoid_tile_init() {
    MATH((SFPU_UNARY_INIT_FN(unused, sfpu::logsigmoid_init, (APPROX, DST_ACCUM_MODE))));
}

}  // namespace ckernel

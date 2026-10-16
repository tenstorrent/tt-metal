// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_prelu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise prelu operation on two inputs: y = idst0 if idst0 is not
 * negative, otherwise idst0 * idst1.
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
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
ALWI void prelu_binary_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_prelu_binary,
        (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void prelu_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_FN(unused, sfpu::calculate_sfpu_prelu_binary_init, (APPROX, DST_ACCUM_MODE))));
}

}  // namespace ckernel

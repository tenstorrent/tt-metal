// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise xlogy operation y = xlogy(x0, x1) with x0 as first operand and x1 as second operand.
 * Output overwrites first operand in DST.
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument              | Description                                                           | Type     | Valid Range                                           | Required |
 * |-----------------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst                  | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void xlogy_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_sfpu_binary_,
        (APPROX, BinaryOp::XLOGY, 8),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void xlogy_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_CB(unused, sfpu::_sfpu_binary_init_, (APPROX, BinaryOp::XLOGY))));
}

}  // namespace ckernel

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_clamp.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise clamp operation for float. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The min value for the clamp function                                       | uint32_t |                                                       | True     |
 * | param1          | The max value for the clamp function                                       | uint32_t |                                                       | True     |
*/
// clang-format on
ALWI void clamp_tile(std::uint32_t idst, std::uint32_t param0, std::uint32_t param1) {
    dest_order::touch_sfpu();
    SFPU(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_clamp,
        (APPROX, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param0,
        param1));
}

#ifndef ARCH_QUASAR
// clang-format off
/**
 * Performs element-wise clamp operation for int32. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The min value for the clamp function                                       | uint32_t |                                                       | True     |
 * | param1          | The max value for the clamp function                                       | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void clamp_tile_int32(std::uint32_t idst, std::uint32_t param0, std::uint32_t param1) {
    dest_order::touch_sfpu();
    SFPU(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_clamp_int32,
        (APPROX, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param0,
        param1));
}
#endif  // !ARCH_QUASAR

/**
 * Please refer to documentation for any_init.
 */
ALWI void clamp_tile_init() { SFPU(SFPU_UNARY_INIT(clamp)); }

}  // namespace ckernel

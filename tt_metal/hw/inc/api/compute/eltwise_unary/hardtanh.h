// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_hardtanh.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Performs element-wise hardtanh operation. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The minimum value of the linear region range                               | uint32_t |                                                       | True     |
 * | param1          | The maximum value of the linear region range                               | uint32_t |                                                       | True     |

 */
// clang-format on
ALWI void hardtanh_tile(std::uint32_t idst, std::uint32_t param0, std::uint32_t param1) {
    dest_order::touch_sfpu();
    SFPU(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_hardtanh,
        (APPROX, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param0,
        param1));
}

ALWI void hardtanh_tile_pack(std::uint32_t idst, std::uint32_t param0, std::uint32_t param1) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_hardtanh,
        (APPROX, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param0,
        param1));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardtanh_tile_init() { SFPU(SFPU_UNARY_INIT(hardtanh)); }

ALWI void hardtanh_tile_init_pack() { PACK(SFPU_UNARY_INIT(hardtanh)); }

}  // namespace ckernel

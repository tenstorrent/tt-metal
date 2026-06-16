// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_snake_beta.h"
#include "llk_math_eltwise_ternary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs elementwise SnakeBeta fused activation: out = x + sin(alpha * x)^2 / beta
 *
 * | Argument   | Description                                                 | Type     | Valid Range                                           | Required |
 * |------------|-------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst_x     | Index of the tile in DST register buffer (input x)         | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_alpha | Index of the tile in DST register buffer (input alpha)     | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_beta  | Index of the tile in DST register buffer (input beta)      | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_out   | Index of the tile in DST register buffer (output)          | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void snake_beta_tile(uint32_t idst_x, uint32_t idst_alpha, uint32_t idst_beta, uint32_t idst_out) {
    MATH((SFPU_TERNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_snake_beta,
        (APPROX, DST_ACCUM_MODE, data_format, 8 /* ITERATIONS */),
        idst_x,
        idst_alpha,
        idst_beta,
        idst_out,
        VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void snake_beta_tile_init() { MATH((SFPU_TERNARY_INIT_FN(snake_beta, sfpu::snake_beta_init, (APPROX)))); }

}  // namespace ckernel

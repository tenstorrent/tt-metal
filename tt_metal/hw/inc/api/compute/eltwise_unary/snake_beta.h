// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_snake_beta.h"
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
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void snake_beta_tile(
    std::uint32_t idst_x, std::uint32_t idst_alpha, std::uint32_t idst_beta, std::uint32_t idst_out) {
    MATH((sfpu::SnakeBeta<APPROX, is_fp32_dest_acc_en, data_format>::run(idst_x, idst_alpha, idst_beta, idst_out)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void snake_beta_tile_init() { MATH((sfpu::SnakeBeta<APPROX, DST_ACCUM_MODE>::init())); }

}  // namespace ckernel

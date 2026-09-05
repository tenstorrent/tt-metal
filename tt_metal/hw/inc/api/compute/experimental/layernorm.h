// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"

#ifdef TRISC_MATH
#include "experimental/llk_math_eltwise_binary_custom_api.h"
#endif
#ifdef TRISC_UNPACK
#include "experimental/llk_unpack_AB_sub_bcast_col_custom_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Prepare a fused, cancellation-resistant LayerNorm subtraction.
 *
 * The statistic tile contains the row anchor in its first column and
 * anchor - mean in its second face column. The operation computes
 * (input - anchor) + (anchor - mean) without rounding the mean to the
 * input magnitude first.
 *
 * Return value: None
 *
 * | Param Type | Name          | Description                                      | Type     | Valid Range | Required |
 * |------------|---------------|--------------------------------------------------|----------|-------------|----------|
 * | Function   | input_cb      | Circular buffer containing input tiles           | uint32_t | 0-31        | True     |
 * | Function   | split_mean_cb | Circular buffer containing split-mean statistics | uint32_t | 0-31        | True     |
 * | Function   | call_line     | Source line reported by state validation          | uint32_t | -           | False    |
 */
// clang-format on
ALWI void sub_bcast_cols_compensated_init(
    std::uint32_t input_cb, std::uint32_t split_mean_cb, std::uint32_t call_line = __builtin_LINE()) {
    state_configure(input_cb, split_mean_cb, call_line);
    MATH((llk_math_sub_bcast_cols_compensated_init()));
    UNPACK((llk_unpack_AB_sub_bcast_col_init_custom(input_cb, split_mean_cb)));
}

// clang-format off
/**
 * Subtract a split mean from consecutive input tiles without cancelling the
 * retained row anchor. Call `sub_bcast_cols_compensated_init` first. The
 * destination range must fit in the acquired DST bank.
 *
 * Return value: None
 *
 * | Param Type | Name          | Description                                                    | Type     | Valid Range | Required |
 * |------------|---------------|----------------------------------------------------------------|----------|-------------|----------|
 * | Function   | input_cb      | Circular buffer containing input tiles                         | uint32_t | 0-31        | True     |
 * | Function   | split_mean_cb | Circular buffer containing anchor and anchor-minus-mean values | uint32_t | 0-31        | True     |
 * | Function   | input_tile    | Index of the first input tile                                  | uint32_t | -           | True     |
 * | Function   | dst_tile      | Index of the first destination tile                            | uint32_t | -           | True     |
 * | Function   | tile_count    | Number of consecutive tiles to process                         | uint32_t | -           | True     |
 */
// clang-format on
ALWI void sub_bcast_cols_compensated(
    std::uint32_t input_cb,
    std::uint32_t split_mean_cb,
    std::uint32_t input_tile,
    std::uint32_t dst_tile,
    std::uint32_t tile_count) {
    MATH((llk_math_sub_bcast_cols_compensated(input_cb, dst_tile, tile_count)));
    UNPACK((llk_unpack_AB_sub_bcast_col_custom(input_cb, split_mean_cb, input_tile, 0 /*tile_index_b*/, tile_count)));
}

}  // namespace ckernel

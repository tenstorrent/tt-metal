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

/**
 * Prepare a fused, cancellation-resistant LayerNorm subtraction.
 *
 * The statistic tile contains the row anchor in its first column and
 * anchor - mean in its second face column. The operation computes
 * (input - anchor) + (anchor - mean) without rounding the mean to the
 * input magnitude first.
 */
ALWI void sub_bcast_cols_compensated_init(
    std::uint32_t input_cb, std::uint32_t split_mean_cb, std::uint32_t call_line = __builtin_LINE()) {
    state_configure(input_cb, split_mean_cb, call_line);
    MATH((llk_math_sub_bcast_cols_compensated_init(input_cb, split_mean_cb)));
    UNPACK((llk_unpack_AB_sub_bcast_col_init_custom(input_cb, split_mean_cb)));
}

ALWI void sub_bcast_cols_compensated(
    std::uint32_t input_cb,
    std::uint32_t split_mean_cb,
    std::uint32_t input_tile,
    std::uint32_t dst_tile,
    std::uint32_t tile_count) {
    MATH((llk_math_sub_bcast_cols_compensated(input_cb, dst_tile, tile_count)));
    UNPACK(
        (llk_unpack_AB_sub_bcast_col_custom(input_cb, split_mean_cb, input_tile, 0 /* split_mean_tile */, tile_count)));
}

}  // namespace ckernel

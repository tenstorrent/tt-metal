// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel_sfpu_where.h"
#include "llk_assert.h"
#include "llk_defs.h"

namespace ckernel {

/**
 * @brief Initializes the SFPU for ternary where operations.
 *
 * Runs the shared SFPU init and Where's own init_kernel (ADDR_MOD_6) via the op struct.
 *
 * @tparam APPROXIMATE  Unused for where; kept for API parity with other SFPU ops.
 */
template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_where_init() {
    sfpu::Where<APPROXIMATE, DataFormat::Float16_b, DST_SYNC_MODE, DST_ACCUM_MODE>::init();
}

/**
 * @brief Executes a ternary per-lane where select over DEST tiles.
 *
 * Dispatches @c calculate_where face-by-face via the @c sfpu::Where op struct.
 * Per-lane result: @c out = (cond != 0) ? true_val : false_val.
 *
 * @tparam APPROXIMATE   Unused for where; kept for API parity with other SFPU ops.
 * @tparam data_format   Unused at call time; kept for API parity with other SFPU ops.
 *
 * @param dst_index0  DEST tile index for the condition operand.
 * @param dst_index1  DEST tile index for the true-branch operand.
 * @param dst_index2  DEST tile index for the false-branch operand.
 * @param odst        DEST tile index that receives the result.
 * @param vector_mode Must be @c VectorMode::RC; Quasar only supports full-tile mode.
 */
template <bool APPROXIMATE, [[maybe_unused]] DataFormat data_format>
inline void llk_math_eltwise_ternary_sfpu_where(
    std::uint32_t dst_index0,
    std::uint32_t dst_index1,
    std::uint32_t dst_index2,
    std::uint32_t odst,
    int vector_mode = (int)VectorMode::RC) {
    LLK_ASSERT(vector_mode == (int)VectorMode::RC, "Quasar currently only supports vector mode RC");
    sfpu::Where<APPROXIMATE, data_format, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
        dst_index0, dst_index1, dst_index2, odst, VectorMode::RC);
}

}  // namespace ckernel

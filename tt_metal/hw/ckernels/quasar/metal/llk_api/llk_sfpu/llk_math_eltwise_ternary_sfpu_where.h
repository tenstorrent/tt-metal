// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_where.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_ternary_sfpu.h"

namespace ckernel {

/**
 * @brief Initializes the SFPU for ternary where operations.
 *
 * Programs shared SFPU state (ADDR_MOD_7) via the common ternary init
 *
 * @tparam APPROXIMATE  Unused for where; kept for API parity with other SFPU ops.
 */
template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_where_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::where>();
}

/**
 * @brief Executes a ternary per-lane where select over DEST tiles.
 *
 * Dispatches @c _calculate_where_ face-by-face via the ternary params wrapper.
 * Per-lane result: @c out = (cond != 0) ? true_val : false_val.
 *
 * @tparam APPROXIMATE   Unused for where; kept for API parity with other SFPU ops.
 * @tparam data_format   Unused at call time; kept for API parity with other SFPU ops.
 *
 * @param dst_index0  DEST tile index for the condition operand.
 * @param dst_index1  DEST tile index for the true-branch operand.
 * @param dst_index2  DEST tile index for the false-branch operand.
 * @param odst        DEST tile index that receives the result.
 * @param vector_mode Faces to process: R (0-1), C (0,2), RC (all 4, default), or scalar (once).
 */
template <bool APPROXIMATE, [[maybe_unused]] DataFormat data_format>
inline void llk_math_eltwise_ternary_sfpu_where(
    std::uint32_t dst_index0,
    std::uint32_t dst_index1,
    std::uint32_t dst_index2,
    std::uint32_t odst,
    VectorMode vector_mode = VectorMode::RC) {
    _llk_math_eltwise_ternary_sfpu_params_(
        sfpu::calculate_where<APPROXIMATE, SFPU_ITERATIONS>, dst_index0, dst_index1, dst_index2, odst, vector_mode);
}

}  // namespace ckernel

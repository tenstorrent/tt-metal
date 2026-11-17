// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_binop_with_unary.h"
#include "llk_assert.h"

namespace ckernel {

/**
 * @brief Initialize SFPU for elementwise unary binop with scalar
 */
inline void llk_math_eltwise_unary_sfpu_binop_with_scalar_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>();
}

/**
 * @brief Performs elementwise unary binop with scalar: y = binop(x, scalar), or scalar - x for Rsub
 *
 * Operates in-place on the DST tile at dst_index.
 *
 * @tparam APPROXIMATE: Approximation mode
 * @tparam binop_mode: Add, Sub, Mul, Div (the host passes 1 / scalar) or Rsub
 * @tparam is_fp32_dest_acc_en: Dest holds fp32; with a 16-bit Dest, Rsub rounds to bf16 nearest-even
 *
 * @param dst_index: The index of the tile in DST register buffer to read and write
 * @param scalar: fp32 scalar value encoded as uint32
 * @param vector_mode: Vector mode (must be VectorMode::RC)
 */
template <bool APPROXIMATE, sfpu::BinopMode binop_mode, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
inline void llk_math_eltwise_unary_sfpu_binop_with_scalar(
    std::uint32_t dst_index, std::uint32_t scalar, VectorMode vector_mode = VectorMode::RC) {
    LLK_ASSERT(vector_mode == VectorMode::RC, "Quasar currently only supports vector mode RC");
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_binop_with_scalar,
        (APPROXIMATE, binop_mode, SFPU_ITERATIONS, is_fp32_dest_acc_en),
        dst_index,
        vector_mode,
        scalar);
}

}  // namespace ckernel

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu.h"
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
 * @brief Performs elementwise unary binop with scalar: y = binop(x, scalar)
 *
 * Operates in-place on the DST tile at dst_index. Quasar currently supports MUL_UNARY (binop_mode = 2)
 * only.
 *
 * @tparam APPROXIMATE: Approximation mode
 * @tparam binop_mode: Binop mode (ADD_UNARY=0, SUB_UNARY=1, MUL_UNARY=2, DIV_UNARY=3, RSUB_UNARY=4)
 *
 * @param dst_index: The index of the tile in DST register buffer to read and write
 * @param scalar: fp32 scalar value encoded as uint32
 * @param vector_mode: Vector mode (must be VectorMode::RC)
 */
template <bool APPROXIMATE, sfpu::BinopMode binop_mode>
inline void llk_math_eltwise_unary_sfpu_binop_with_scalar(
    std::uint32_t dst_index, std::uint32_t scalar, VectorMode vector_mode = VectorMode::RC) {
    LLK_ASSERT(vector_mode == VectorMode::RC, "Quasar currently only supports vector mode RC");
    static_assert(binop_mode == sfpu::BinopMode::Mul, "Quasar binop_with_scalar currently supports Mul only");
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::calculate_binop_with_scalar<APPROXIMATE, binop_mode, SFPU_ITERATIONS>,
        dst_index,
        vector_mode,
        scalar);
}

}  // namespace ckernel

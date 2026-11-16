// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_assert.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_ema.h"

namespace ckernel {

/**
 * @brief Configure the math thread for column-wise EMA; then load the weights and clear the carry.
 *
 * @note Another SFPU op between EMA tiles ends the chain (it may clobber LREG4-6): redo all three
 *       calls. FPU ops between EMA tiles need none of them.
 */
inline void llk_math_ema_sfpu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(sfpu::init_ema<sfpu::EMA_OUTPUT_TILE_DELTA>);
}

/**
 * @brief Load the fp32 weights (alpha on the carry, beta on the input); they stay resident.
 */
inline void llk_math_ema_sfpu_load_alpha_beta(const std::uint32_t alpha, const std::uint32_t beta) {
    sfpu::ema_load_alpha_beta(alpha, beta);
}

/**
 * @brief Zero the carry; call once per chain, not per tile.
 */
inline void llk_math_ema_sfpu_clear_previous_output() { sfpu::ema_clear_previous_output(); }

/**
 * @brief Column-wise EMA of Dest tile input_dst_index into tile input_dst_index + 1.
 *
 * @note Feed tiles top to bottom: each call continues the previous call's carry.
 */
inline void llk_math_ema_sfpu_tile(const std::uint32_t input_dst_index) {
    // SFPU_UNARY_CALL only bounds-checks the input tile.
    constexpr std::uint32_t max_dest_tiles =
        trisc::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, trisc::DstTileShape::Tile32x32>();
    LLK_ASSERT(
        input_dst_index + sfpu::EMA_OUTPUT_TILE_DELTA < max_dest_tiles,
        "ema_tile: output tile (input_dst_index + 1) must fit in Dest");
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_ema, input_dst_index, VectorMode::RC_custom);
}

}  // namespace ckernel

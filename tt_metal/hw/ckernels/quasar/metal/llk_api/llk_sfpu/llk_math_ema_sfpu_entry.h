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
 * @brief Configure the math thread for column-wise EMA.
 *
 * @note Follow this with @ref llk_math_ema_sfpu_load_alpha_beta and
 *       @ref llk_math_ema_sfpu_clear_previous_output before the first @ref llk_math_ema_sfpu_tile.
 * @note Another SFPU op between EMA tiles ends the chain: it may overwrite the weights and the
 *       carry (LREG4-6), and the carry cannot be restored. Run all three calls again to start a new
 *       chain. FPU ops between EMA tiles need none of them.
 */
inline void llk_math_ema_sfpu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(sfpu::init_ema<sfpu::EMA_OUTPUT_TILE_DELTA>);
}

/**
 * @brief Install the EMA smoothing weights for every later tile.
 *
 * @param alpha: fp32 bit pattern of the weight on the carry (EMA_old).
 * @param beta: fp32 bit pattern of the weight on the incoming datum.
 * @note Call after @ref llk_math_ema_sfpu_init; the weights stay resident across tiles.
 */
inline void llk_math_ema_sfpu_load_alpha_beta(const std::uint32_t alpha, const std::uint32_t beta) {
    sfpu::ema_load_alpha_beta(alpha, beta);
}

/**
 * @brief Zero the running EMA carry, starting a fresh top-to-bottom chain.
 *
 * @note Call once before the tile that begins a chain, not per tile - consecutive
 *       @ref llk_math_ema_sfpu_tile calls are meant to continue one time sequence.
 */
inline void llk_math_ema_sfpu_clear_previous_output() { sfpu::ema_clear_previous_output(); }

/**
 * @brief Apply column-wise EMA to one Dest tile, writing the result to the next tile.
 *
 * @param input_dst_index: Tile index of the input in the destination register; the output lands in
 *        tile input_dst_index + 1.
 * @note Call @ref llk_math_ema_sfpu_init and @ref llk_math_ema_sfpu_load_alpha_beta before this.
 *       Feed tiles top-to-bottom: the carry from this call is what the next one continues from.
 */
inline void llk_math_ema_sfpu_tile(const std::uint32_t input_dst_index) {
    // SFPU_UNARY_CALL bounds-checks the input tile; the output lands one tile further on.
    constexpr std::uint32_t max_dest_tiles =
        trisc::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, trisc::DstTileShape::Tile32x32>();
    LLK_ASSERT(
        input_dst_index + sfpu::EMA_OUTPUT_TILE_DELTA < max_dest_tiles,
        "ema_tile: output tile (input_dst_index + 1) must fit in Dest");
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_ema, input_dst_index, VectorMode::RC_custom);
}

}  // namespace ckernel

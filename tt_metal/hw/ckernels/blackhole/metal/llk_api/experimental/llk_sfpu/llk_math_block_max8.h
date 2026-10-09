// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sanitizer/api.h"
#include "sfpu/experimental/ckernel_sfpu_block_max8.h"

namespace ckernel {

/**
 * @brief Initialize standard SFPU configuration and address modifiers for BF16 block max.
 *
 * SfpuType::unused selects generic initialization: this callback uses ADDR_MOD_7
 * and needs no operation-specific address modifier configuration.
 * @tparam FP32: Must be false; this operation supports BF16 DST only.
 * @note Use BF16 DST with SyncHalf. Call before @ref llk_math_block_max8.
 */
template <bool FP32>
inline void llk_math_block_max8_init() {
    static_assert(!FP32, "block_max8 supports BF16 DEST only");
    static_assert(DST_SYNC_MODE == DstSync::SyncHalf, "block_max8 supports SyncHalf only");
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, FP32>();
}

/**
 * @brief Dispatch one in-place block-max operation through the standard unary SFPU LLK.
 *
 * RC_custom invokes the ckernel implementation once for all four faces and
 * compaction. The dispatcher anchors addresses at dst_index and owns normal
 * SFPU start/done synchronization; no custom SFPU math LLK is required.
 * @tparam FP32: Must be false; BF16 DST only.
 * @param dst_index: Runtime tile index in the acquired BF16 DST half.
 * @param valid_scores: Length of the valid row-major prefix, from zero through 1024.
 * @note Call @ref llk_math_block_max8_init first. Keep runtime DST accumulation
 *       in BF16 mode; dynamic FP32 DST is unsupported. Inputs outside the stated
 *       ranges are unsupported and are diagnosed only when LLK assertions are enabled.
 */
template <bool FP32>
inline void llk_math_block_max8(uint32_t dst_index, uint32_t valid_scores) {
    static_assert(!FP32, "block_max8 supports BF16 DEST only");
    static_assert(DST_SYNC_MODE == DstSync::SyncHalf, "block_max8 supports SyncHalf only");
    SAN_HOOK(unsupported());
    LLK_ASSERT(
        (get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>() == DEST_NUM_TILES_FP16_HALF),
        "block_max8 requires runtime BF16 DST accumulation");
    LLK_ASSERT(dst_index < DEST_NUM_TILES_FP16_HALF, "DST index exceeds the acquired BF16 half");
    LLK_ASSERT(valid_scores <= TILE_R_DIM * TILE_C_DIM, "Valid prefix exceeds one tile");
    _llk_math_eltwise_unary_sfpu_params_(sfpu::_calculate_block_max8_, dst_index, VectorMode::RC_custom, valid_scores);
}

}  // namespace ckernel

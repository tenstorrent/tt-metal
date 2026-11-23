// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "experimental/llk_math_block_max8_fpu.h"
#include "llk_math_common_api.h"
#include "sanitizer/api.h"

namespace ckernel {

/**
 * @brief Initialize the FPU (GMPOOL) block max for BF16 DST.
 * @tparam FP32: Must be false; this operation supports BF16 DST only.
 * @note Use BF16 DST with SyncHalf. Call before @ref llk_math_block_max8_fpu.
 */
template <bool FP32>
inline void llk_math_block_max8_fpu_init() {
    static_assert(!FP32, "block_max8_fpu supports BF16 DEST only");
    static_assert(DST_SYNC_MODE == DstSync::SyncHalf, "block_max8_fpu supports SyncHalf only");
    _llk_math_block_max8_fpu_init_();
}

/**
 * @brief Pool one DST tile in place with GMPOOL: 1024 scores to 128 block maxima in the first eight rows.
 * @tparam FP32: Must be false; BF16 DST only.
 * @param dst_index: Runtime tile index in the acquired BF16 DST half.
 * @param valid_scores: Length of the valid row-major prefix, from zero through 1024.
 * @note Call @ref llk_math_block_max8_fpu_init first. The unpack thread must call
 *       @ref llk_unpack_block_max8_fpu once for each call.
 */
template <bool FP32>
inline void llk_math_block_max8_fpu(uint32_t dst_index, uint32_t valid_scores) {
    static_assert(!FP32, "block_max8_fpu supports BF16 DEST only");
    static_assert(DST_SYNC_MODE == DstSync::SyncHalf, "block_max8_fpu supports SyncHalf only");
    SAN_HOOK(unsupported());
    LLK_ASSERT(
        (get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>() == DEST_NUM_TILES_FP16_HALF),
        "block_max8_fpu requires runtime BF16 DST accumulation");
    LLK_ASSERT(dst_index < DEST_NUM_TILES_FP16_HALF, "DST index exceeds the acquired BF16 half");
    LLK_ASSERT(valid_scores <= TILE_R_DIM * TILE_C_DIM, "Valid prefix exceeds one tile");
    _llk_math_block_max8_fpu_(dst_index, valid_scores);
}

}  // namespace ckernel

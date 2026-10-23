// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_pack_tile_api.h"

/**
 * @brief Initializes the packer for @ref llk_pack_block_contiguous.
 *
 * @param output Logical output DataFlow Buffer identifier
 * @note Call again when the output DFB or tile shape changes, not when only num_tiles changes. Pack hardware
 *       configuration must precede this.
 */
inline void llk_pack_block_contiguous_mop_config(const std::uint32_t output) { llk_pack_init(output); }

/**
 * @brief Packs num_tiles consecutive DEST tiles, starting at tile_index, into the output DFB in one MOP launch.
 *
 * @tparam EN_32BIT_DEST Must match DST_ACCUM_MODE
 * @param tile_index Starting DEST tile index
 * @param output Logical output DataFlow Buffer identifier
 * @param num_tiles Number of tiles to pack, at least 1
 * @note Call @ref llk_pack_block_contiguous_mop_config first. The output DFB must have stride_size_tiles == 1.
 */
template <bool EN_32BIT_DEST>
inline void llk_pack_block_contiguous(
    const std::uint32_t tile_index, const std::uint32_t output, const std::uint32_t num_tiles) {
    static_assert(EN_32BIT_DEST == DST_ACCUM_MODE, "Pack DEST mode must match the configured accumulation mode");
    LLK_ASSERT(num_tiles >= 1, "Contiguous pack requires at least one tile");
    LLK_ASSERT(
        get_local_dfb_interface(get_output_id(output)).stride_size_tiles == 1,
        "Contiguous pack requires consecutive output entries");
    llk_pack_block(tile_index, output, num_tiles);
}

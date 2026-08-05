// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_pack_common_api.h"
#include "llk_pack_untilize.h"

/*************************************************************************
 * LLK PACK UNTILIZE
 *************************************************************************/

/**
 * Initialize the packer for an untilize operation on the given output operand.
 *
 * Face geometry (face_r_dim, num_faces) is derived from the output CB metadata. In
 * debug builds, validates that the packers are configured correctly for the resolved
 * face row dimension before programming the untilize init sequence.
 *
 * @tparam block_ct_dim   Width of a single block in tiles.
 * @tparam full_ct_dim    Width of the full input in tiles (defaults to block_ct_dim).
 * @tparam diagonal       Whether to use diagonal packing.
 * @tparam narrow_row     Whether the input rows are narrow.
 * @tparam row_num_datums Number of datums per row.
 * @tparam dense          Pack two 2-face tiles into a single 4-face region (unused on Wormhole; must be false).
 * @param  output         Output circular buffer / operand index.
 */
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false>
inline void llk_pack_untilize_init(std::uint32_t output) {
    static_assert(dense == false, "Dense is only supported on BH");
    const std::uint32_t output_id = get_output_id(output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    _llk_pack_untilize_init_<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(
        pack_dst_format[output_id], face_r_dim, num_faces);
}

/**
 * Pack an untilized block of tiles from the destination register into the output CB.
 *
 * Iterates over block_rt_dim tile rows, computing the packer write address from the
 * output CB fifo state for each row. Face geometry (face_r_dim, num_faces) is derived
 * from the output CB metadata.
 *
 * @tparam block_ct_dim       Width of a single block in tiles.
 * @tparam full_ct_dim        Width of the full input in tiles (defaults to block_ct_dim).
 * @tparam diagonal           Whether to use diagonal packing.
 * @tparam narrow_row         Whether the input rows are narrow.
 * @tparam row_num_datums     Number of datums per row.
 * @tparam tile_dst_ct_offset Compile-time column offset of the tile in the destination register.
 * @tparam dense              Pack two 2-face tiles into a single 4-face region (unused on Wormhole; must be false).
 * @param  block_rt_dim       Height of the block in tiles (number of rows to pack).
 * @param  output             Output circular buffer / operand index.
 * @param  block_c_index      Block column index (used when full_ct_dim > block_ct_dim).
 * @param  tile_dst_rt_offset Runtime row offset of the tile in the destination register.
 */
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    std::uint32_t tile_dst_ct_offset = 0,
    bool dense = false>
inline void llk_pack_untilize(
    std::uint32_t block_rt_dim,
    std::uint32_t output,
    const std::uint32_t block_c_index = 0,
    const std::uint32_t tile_dst_rt_offset = 0) {
    static_assert(dense == false, "Dense is only supported on BH");
    const std::uint32_t output_id = get_output_id(output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    std::uint32_t pack_tile_addr =
        get_local_cb_interface(output_id).fifo_wr_ptr - 1 +
        SCALE_DATUM_SIZE(
            pack_dst_format[output_id],
            (block_c_index * ((num_faces > 2) ? num_faces / 2 : num_faces) * block_ct_dim * FACE_C_DIM)) /
            16;

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    for (std::uint32_t block_rt = 0; block_rt < block_rt_dim; block_rt++) {
        _llk_pack_untilize_<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums, tile_dst_ct_offset>(
            pack_tile_addr, pack_dst_format[output_id], face_r_dim, block_rt * block_ct_dim + tile_dst_rt_offset);

        pack_tile_addr += full_ct_dim * get_local_cb_interface(output_id).fifo_page_size;
    }
}

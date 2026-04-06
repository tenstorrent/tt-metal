// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "chlkc_list.h"
#include "internal/circular_buffer_interface.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cpack_common.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_outputs.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_pack_untilize.h"
#include "experimental/dataflow_buffer.h"

/*************************************************************************
 * LLK PACK UNTILIZE
 *************************************************************************/

/**
 * @brief Initializes the packer MOP and hardware stride registers for pack untilize based on the tile shape and tensor
 * dimensions. Must be called before llk_pack_untilize.
 *
 * @tparam block_ct_dim  Width of a single block in tiles.
 * @tparam full_ct_dim   Total width of the tensor row in tiles (default = block_ct_dim).
 * @param pack_output    Output circular buffer identifier.
 */
template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim = block_ct_dim>
inline void llk_pack_untilize_init(std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);

    const TileShape output_tile_shape = {
        .num_faces = get_output_num_faces(output_id),
        .face_r_dim = get_output_face_r_dim(output_id),
        .face_c_dim = FACE_C_DIM,
        .narrow_tile = static_cast<bool>(get_output_narrow_tile(output_id)),
    };

    // TODO: Once narrow-tile is supported c_dim_faces will be variable.
    // For now we set it to 2, instead of deducing a template parameter at runtime
    constexpr std::uint32_t c_dim_faces = 2; //c_dim_faces is 2 (standard 32x32 tile) and 1 for narrow tile where y-dim is <32

    _llk_pack_untilize_init_<full_ct_dim, block_ct_dim, c_dim_faces>(output_id, output_tile_shape);
}

/**
 * @brief Reads block_rt_dim tile-rows out of the dst register and writes them to L1 in
 *        untilized (row-major) layout. For the SyncHalf path (pack_untilize_block), call with block_rt_dim=1 and zero
 *        offsets, the loop executes once and dest_idx is 0 since each tile-row occupies the current dst bank from
 * position 0.
 *
 *        For the pre-filled dst path (pack_untilize_dest), call with block_rt_dim > 1 and the appropriate offsets
 *        dest_idx advances per row to reach tiles that were placed at non-zero positions by a prior math op.
 *
 * @tparam block_ct_dim       Width of a single block in tiles.
 * @tparam full_ct_dim        Total width of the tensor row in tiles, used to compute the
 *                            L1 row stride (default = block_ct_dim).
 * @param block_rt_dim        Number of tile-rows to pack.
 * @param pack_output         Output circular buffer identifier.
 * @param block_c_index       Column-block index within the full row, used to offset the L1
 *                            write address when full_ct_dim > block_ct_dim (default 0).
 */
template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim = block_ct_dim>
inline void llk_pack_untilize(
    std::uint32_t block_rt_dim,
    std::uint32_t pack_output,
    const std::uint32_t block_c_index = 0,
    const std::uint32_t tile_dst_rt_offset = 0) {
    const std::uint32_t output_id = get_output_id(pack_output);

    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);
    const std::uint32_t C_DIM_FACES = narrow_tile ? 1 : 2;
    const std::uint32_t R_DIM_FACES = (num_faces == 2 && !narrow_tile) ? 1 : 2;

    // Each tile is packed in two 16x32 halves — top faces (0+1) then bottom faces (2+3)
    // merging adjacent face-columns into a single output row. Hence we use R_DIM_FACES instead of num_faces for L1
    // strides
    const std::uint32_t y_stride = full_ct_dim * R_DIM_FACES * face_r_dim;
    const std::uint32_t base_l1 = g_dfb_interface[output_id].wr_entry_idx * R_DIM_FACES * face_r_dim;

    for (std::uint32_t block_rt = 0; block_rt < block_rt_dim; block_rt++) {
        const std::uint32_t dest_idx = block_rt * block_ct_dim + tile_dst_rt_offset;
        const std::uint32_t l1_tile_idx = base_l1 + block_rt * y_stride + block_c_index * block_ct_dim;
        _llk_pack_untilize_(dest_idx, l1_tile_idx);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_io.h"
#include "llk_outputs.h"
#include "experimental/llk_pack_block.h"

/*************************************************************************
 * LLK PACK BLOCK CONTIGUOUS API
 *
 * Packs multiple tiles from sparse DEST (Tile32x32 slot convention)
 * to dense L1 (contiguous output) in a single call.
 *
 * Usage:
 *   1. llk_pack_init (or equivalent) — sets addr_mods, strides
 *   2. llk_pack_block_contiguous_mop_config — programs REPLAY + MOP
 *      (only when tile dims change, NOT when num_tiles changes)
 *   3. llk_pack_block_contiguous — packs num_tiles tiles in one call
 *      (num_tiles can vary per call with no reconfig needed)
 *************************************************************************/

// Program the REPLAY buffer and MOP for block-contiguous packing.
// Call once (or when tile dimensions change). Does NOT need to be
// called again when only num_tiles changes between pack calls.
template <bool zero_output = false>
inline void llk_pack_block_contiguous_mop_config(const std::uint32_t output) {
    const std::uint32_t output_id = get_output_id(output);
    _llk_pack_block_contiguous_mop_config_<zero_output>(
        pack_dst_format[output_id], get_output_face_r_dim(output_id), get_output_num_faces(output_id));
}

// Pack num_tiles tiles from sparse DEST to dense L1 in a single call.
// tile_index: starting DEST tile slot (typically 0).
// output: CB identifier.
// num_tiles: number of tiles to pack (1-8, runtime parameter).
template <bool is_fp32_dest_acc_en>
inline void llk_pack_block_contiguous(std::uint32_t tile_index, std::uint32_t output, std::uint32_t num_tiles) {
    std::uint8_t output_id = get_output_id(output);
    std::uint32_t pack_tile_addr =
        get_local_cb_interface(output_id).fifo_wr_ptr + get_local_cb_interface(output_id).fifo_wr_tile_ptr - 1;

    // Advance the CB write pointer by num_tiles pages (so subsequent
    // pack calls or cb_push_back see the correct position).
    get_local_cb_interface(output_id).fifo_wr_tile_ptr += get_local_cb_interface(output_id).fifo_page_size * num_tiles;

    _llk_pack_block_contiguous_<DST_SYNC_MODE, is_fp32_dest_acc_en>(tile_index, pack_tile_addr, num_tiles);
}

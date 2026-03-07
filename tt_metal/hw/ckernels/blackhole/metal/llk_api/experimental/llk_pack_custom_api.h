// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "experimental/llk_pack_custom.h"

/*************************************************************************
 * LLK PACK
 *************************************************************************/

// TODO NC: Remove as the part of tt-metal#34499

// WARNING: Experimental API for SDPA optimizations only.
// This header has no corresponding tests in the llk-test infrastructure.
// Do not use outside of SDPA optimization workflows.

template <bool untilize = false, bool zero_output = false, bool tilize = false>
inline void llk_pack_mop_config_custom(const uint32_t output, std::uint32_t num_tiles = 1) {
    const std::uint32_t output_id = get_output_id(output);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t tile_c_dim = get_output_tile_c_dim(output_id);
    const bool partial_face = get_output_partial_face(output_id) && IS_BFP_FORMAT((uint)pack_dst_format[output_id]);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    _llk_pack_mop_config_custom_<untilize, zero_output, tilize>(
        pack_dst_format[output_id], face_r_dim, tile_c_dim, num_faces, partial_face, narrow_tile, num_tiles);
}

template <bool out_of_order_output, bool is_fp32_dest_acc_en>
inline void llk_pack_w_acc_custom(
    std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0, std::uint32_t num_tiles = 1) {
    std::uint8_t output_id = get_output_id(output);

    std::uint32_t pack_tile_addr = get_output_tile_address<out_of_order_output, false>(output_id, output_tile_index);

    LLK_ASSERT((tile_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");
    _llk_pack_w_acc_custom_<out_of_order_output, is_fp32_dest_acc_en>(tile_index, pack_tile_addr, num_tiles);
}

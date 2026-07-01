// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"
#include "llk_pack_tilize_dispatch.h"

/*************************************************************************
 * LLK PACK
 *************************************************************************/

template <
    PackMode pack_mode = PackMode::Default,
    bool zero_output = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false>
inline void llk_pack_init(
    const std::uint32_t pack_output = 16, std::uint32_t num_tiles = 1, const std::uint32_t input_operand = 0) {
    // TODO (https://github.com/tenstorrent/tt-metal/issues/18948): Revisit for narrow_tile
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t tile_c_dim = get_output_tile_c_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    if constexpr (!skip_addrmod_config) {
        LLK_ASSERT_BLOCK(are_packers_configured_correctly<PackerProgramType::ProgramByFace>(
            pack_src_format[output_id], pack_dst_format[output_id], face_r_dim));
    }

    // When caller requested Tilize but the BH unpack stride bug does not apply
    // (8-bit input), fall back to Default. See unpack_tilize_interleaves_rows
    // in llk_pack_tilize_dispatch.h for the HW background.
    const std::uint32_t src_format = static_cast<std::uint32_t>(unpack_src_format[input_operand]);
    if (pack_mode == PackMode::Tilize && !unpack_tilize_interleaves_rows(src_format)) {
        _llk_pack_init_<PackMode::Default, zero_output, skip_addrmod_config, skip_packer_strides>(
            pack_src_format[output_id], face_r_dim, tile_c_dim, num_faces, num_tiles);
    } else {
        _llk_pack_init_<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
            pack_src_format[output_id], face_r_dim, tile_c_dim, num_faces, num_tiles);
    }
}

template <bool is_fp32_dest_acc_en, bool out_of_order_output = false, PackMode pack_mode = PackMode::Default>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);

    static_assert(
        !((pack_mode == PackMode::Untilize) && out_of_order_output), "untilize out of order packing is not supported!");

    std::uint32_t pack_tile_addr =
        get_output_tile_address<out_of_order_output, pack_mode>(output_id, output_tile_index);

    LLK_ASSERT_BLOCK(are_packers_configured_correctly<PackerProgramType::ProgramByFace>(
        pack_src_format[output_id], pack_dst_format[output_id], get_output_face_r_dim(output)));

    LLK_ASSERT(
        (tile_index < get_pack_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");
    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode>(tile_index, pack_tile_addr);
}

template <bool is_fp32_dest_acc_en, bool out_of_order_output = false, PackMode pack_mode = PackMode::Default>
inline void llk_matmul_pack(
    std::uint32_t start_tile_index, std::uint32_t output, uint32_t ntiles, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);

    static_assert(
        !((pack_mode == PackMode::Untilize) && out_of_order_output), "untilize out of order packing is not supported!");
    LLK_ASSERT_BLOCK(are_packers_configured_correctly<PackerProgramType::ProgramByFace>(
        pack_src_format[output_id], pack_dst_format[output_id], get_output_face_r_dim(output)));
    LLK_ASSERT(
        ((start_tile_index + ntiles - 1) < get_pack_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");

    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        std::uint32_t pack_tile_addr =
            get_output_tile_address<out_of_order_output, pack_mode>(output_id, output_tile_index);

        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode>(tile_index, pack_tile_addr);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"
#include "llk_pack_untilize.h"
#include "llk_param_structs.h"

/*************************************************************************
 * LLK PACK UNTILIZE
 *************************************************************************/

template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void llk_pack_untilize_hw_configure(const llk_pack_params_t* pack_params) {
    const std::uint32_t output_id = get_output_id(pack_params->pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const std::uint32_t tile_c_dim = get_output_tile_c_dim(output_id);
    const bool partial_face = get_output_partial_face(output_id);

    const std::uint32_t tile_size = get_local_cb_interface(output_id).fifo_page_size;

    _llk_pack_hw_configure_<is_fp32_dest_acc_en, pack_mode>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        tile_size,
        face_r_dim,
        tile_c_dim,
        num_faces,
        partial_face,
        pack_params->relu_config.val);
}

template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false /* unused */,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false>
inline void llk_pack_untilize_init(std::uint32_t output) {
    static_assert(diagonal == false, "Diagonal is only supported on WH");
    const std::uint32_t output_id = get_output_id(output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    LLK_ASSERT_BLOCK(are_packers_configured_correctly<PackerProgramType::ProgramByFace>(
        pack_src_format[output_id], pack_dst_format[output_id], face_r_dim));

    _llk_pack_untilize_init_<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense>(
        pack_src_format[output_id], pack_dst_format[output_id], face_r_dim, num_faces);
}

inline void llk_pack_untilize_uninit(std::uint32_t output) {
    const std::uint32_t output_id = get_output_id(output);
    _llk_pack_untilize_uninit_(pack_src_format[output_id]);
}

template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false /* unused */,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM /* unused */,
    uint32_t tile_dst_ct_offset = 0,
    bool dense = false>
inline void llk_pack_untilize(
    std::uint32_t block_rt_dim,
    std::uint32_t output,
    const std::uint32_t block_c_index = 0,
    const std::uint32_t tile_dst_rt_offset = 0) {
    static_assert(diagonal == false, "Diagonal is only supported on WH");
    const std::uint32_t output_id = get_output_id(output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    std::uint32_t pack_tile_addr =
        get_local_cb_interface(output_id).fifo_wr_ptr - 1 +
        SCALE_DATUM_SIZE(
            pack_dst_format[output_id],
            (block_c_index * ((num_faces > 2) ? num_faces / 2 : num_faces) * block_ct_dim * FACE_C_DIM)) /
            16;

    LLK_ASSERT_BLOCK(are_packers_configured_correctly<PackerProgramType::ProgramByFace>(
        pack_src_format[output_id], pack_dst_format[output_id], face_r_dim));

    for (std::uint32_t block_rt = 0; block_rt < block_rt_dim; block_rt++) {
        _llk_pack_untilize_<block_ct_dim, full_ct_dim, narrow_row, tile_dst_ct_offset, dense>(
            pack_tile_addr, num_faces, block_rt * block_ct_dim + tile_dst_rt_offset);

        pack_tile_addr += full_ct_dim * get_local_cb_interface(output_id).fifo_page_size;
    }
}

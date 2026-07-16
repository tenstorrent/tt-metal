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

/**
 * Configure the packer hardware for an untilize output operand.
 *
 * Face geometry (face_r_dim, num_faces), tile column dimension, partial-face flag
 * and tile size are all derived from the output CB metadata associated with the
 * operand id. Callers no longer thread face geometry through the API, since per-CB
 * face geometry is recorded in the CB descriptor at program creation time. The relu
 * configuration is taken from the supplied pack params.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam pack_mode           Packer program mode (e.g. Default, Untilize).
 * @param  pack_params         Pack parameters carrying the output operand and relu config.
 */
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

/**
 * Initialize the packer for an untilize operation on the given output operand.
 *
 * Face geometry (face_r_dim, num_faces) is derived from the output CB metadata. In
 * debug builds, validates that the packers are configured correctly for the resolved
 * face row dimension before programming the untilize init sequence.
 *
 * @tparam block_ct_dim   Width of a single block in tiles.
 * @tparam full_ct_dim    Width of the full input in tiles (defaults to block_ct_dim).
 * @tparam diagonal       Diagonal packing flag (unused on Blackhole; must be false).
 * @tparam narrow_row     Whether the input rows are narrow.
 * @tparam row_num_datums Number of datums per row.
 * @tparam dense          Pack two 2-face tiles into a single 4-face region.
 * @param  output         Output circular buffer / operand index.
 */
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

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    _llk_pack_untilize_init_<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense>(
        pack_src_format[output_id], pack_dst_format[output_id], face_r_dim, num_faces);
}

/**
 * Tear down the packer untilize configuration so a subsequent operation can
 * reprogram the packer. The source format is read from the output CB metadata.
 *
 * @param output Output circular buffer / operand index.
 */
inline void llk_pack_untilize_uninit(std::uint32_t output) {
    const std::uint32_t output_id = get_output_id(output);
    _llk_pack_untilize_uninit_(pack_src_format[output_id]);
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
 * @tparam diagonal           Diagonal packing flag (unused on Blackhole; must be false).
 * @tparam narrow_row         Whether the input rows are narrow.
 * @tparam row_num_datums     Number of datums per row (unused on Blackhole).
 * @tparam tile_dst_ct_offset Compile-time column offset of the tile in the destination register.
 * @tparam dense              Pack two 2-face tiles into a single 4-face region.
 * @param  block_rt_dim       Height of the block in tiles (number of rows to pack).
 * @param  output             Output circular buffer / operand index.
 * @param  block_c_index      Block column index (used when full_ct_dim > block_ct_dim).
 * @param  tile_dst_rt_offset Runtime row offset of the tile in the destination register.
 */
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

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    for (std::uint32_t block_rt = 0; block_rt < block_rt_dim; block_rt++) {
        _llk_pack_untilize_<block_ct_dim, full_ct_dim, narrow_row, tile_dst_ct_offset, dense>(
            pack_tile_addr, num_faces, block_rt * block_ct_dim + tile_dst_rt_offset);

        pack_tile_addr += full_ct_dim * get_local_cb_interface(output_id).fifo_page_size;
    }
}

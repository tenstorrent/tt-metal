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
 * @brief Configures the packer buffer descriptor and dst register format for pack untilize.
 *        Sets the dimensions in the buffer descriptor to the untilize layout (face_c_dim as
 *        x_dim, face_r_dim as y_dim) and programs the dst register format from the pack
 *        source format. Must be called before llk_pack_untilize_init.
 *
 * @param pack_output  Output buffer identifier.
 */
inline void llk_pack_untilize_hw_configure_disaggregated(std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    // Configure buffer descriptor with the untilize face dimensions
    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B = g_dfb_interface[output_id].tc_slots[0].base_addr;
    bd_val.f.format = static_cast<std::uint8_t>(pack_dst_format[output_id]);
    bd_val.f.x_dim = ckernel::trisc::FACE_C_DIM;
    bd_val.f.y_dim = face_r_dim;
    bd_val.f.z_dim = num_faces;
    ckernel::trisc::_configure_buf_desc_table_(output_id, bd_val);

    // Configure the dst register format
    tdma_descriptor_t td_val{};
    td_val.reg_data_format = static_cast<std::uint8_t>(pack_src_format[output_id]);
    _llk_pack_hw_configure_<p_pacr::PACK0>(td_val);
}

/**
 * @brief Initializes the packer MOP and hardware stride registers for pack untilize based on the tile shape and tensor
 * dimensions. Must be called after llk_pack_untilize_hw_configure_disaggregated and before llk_pack_untilize.
 *
 * @tparam block_ct_dim  Width of a single block in tiles.
 * @tparam full_ct_dim   Total width of the tensor row in tiles (default = block_ct_dim).
 * @param output         Output circular buffer identifier.
 */
template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim = block_ct_dim>
inline void llk_pack_untilize_init(std::uint32_t output) {
    const std::uint32_t output_id = get_output_id(output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    const TileShape tile_shape = {
        .num_faces = num_faces,
        .face_r_dim = face_r_dim,
        .face_c_dim = ckernel::trisc::FACE_C_DIM,
        .narrow_tile = narrow_tile,
    };

    constexpr std::uint32_t c_dim_faces = 2;

    _llk_pack_untilize_init_<full_ct_dim, block_ct_dim, c_dim_faces>(output_id, tile_shape);
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
 * @param output              Output circular buffer identifier.
 * @param block_c_index       Column-block index within the full row, used to offset the L1
 *                            write address when full_ct_dim > block_ct_dim (default 0).
 */
template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim = block_ct_dim>
inline void llk_pack_untilize(std::uint32_t block_rt_dim, std::uint32_t output, const std::uint32_t block_c_index = 0) {
    const std::uint32_t output_id = get_output_id(output);

    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);
    const std::uint32_t C_DIM_FACES = narrow_tile ? 1 : 2;
    const std::uint32_t R_DIM_FACES = (num_faces == 2 && !narrow_tile) ? 1 : 2;

    const std::uint32_t base_l1 = g_dfb_interface[output_id].wr_entry_idx;
    const std::uint32_t y_stride = full_ct_dim * R_DIM_FACES * face_r_dim;
    // std::uint32_t y_stride_external = full_ct_dim * R_DIM_FACES * face_r_dim;
    // _llk_pack_untilize_(0, block_c_index * y_stride_external);
    _llk_pack_untilize_(0, base_l1 + 0 * y_stride + block_c_index * block_ct_dim);
}

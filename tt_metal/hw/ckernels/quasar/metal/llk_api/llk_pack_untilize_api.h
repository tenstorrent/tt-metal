// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "chlkc_list.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "tensor_shape.h"
#include "ckernel_template.h"
#include "cpack_common.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_outputs.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_pack_untilize.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK PACK UNTILIZE
 *************************************************************************/

/**
 * @brief Initializes the packer MOP and hardware stride registers for pack untilize based on the tensor shape and
 * tensor dimensions. Must be called before llk_pack_untilize.
 *
 * @tparam block_ct_dim  Width of a single block in tiles.
 * @tparam full_ct_dim   Total width of the tensor row in tiles (default = block_ct_dim).
 * @param pack_output    Output DataFlow Buffer identifier.
 */
template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim = block_ct_dim>
inline void llk_pack_untilize_init(std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);

    const ckernel::TensorShape tensor_shape = get_output_tensor_shape(output_id);

    if (tensor_shape.face_r_dim < ckernel::pack::PACR_STRIDE_OFFSET_ROWS) {
        const tdma_descriptor_t td_val = ckernel::trisc::construct_tdma_desc<ckernel::trisc::L1AccessMode::Strided>(
            tensor_shape,
            get_local_dfb_interface(output_id).tc_slots[0].base_addr,
            pack_dst_format[output_id],
            output_id,
            pack_src_format[output_id]);
        ckernel::trisc::_configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
    }

    _llk_pack_untilize_init_<full_ct_dim, block_ct_dim>(output_id, tensor_shape);
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
 * @param pack_output         Output DataFlow Buffer identifier.
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

    const ckernel::TensorShape tensor_shape = get_output_tensor_shape(output_id);
    // Each tile is packed in two 16x32 halves — top faces (0+1) then bottom faces (2+3)
    // merging adjacent face-columns into a single output row. Hence we use num_faces_r_dim instead of num_faces for L1
    // strides

    const std::uint32_t y_stride = full_ct_dim * tensor_shape.num_faces_r_dim * tensor_shape.face_r_dim;
    const LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(output_id);
    const std::uint32_t base_l1 = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].wr_entry_idx *
                                  tensor_shape.num_faces_r_dim * tensor_shape.face_r_dim;

    for (std::uint32_t block_rt = 0; block_rt < block_rt_dim; block_rt++) {
        const std::uint32_t dest_idx = block_rt * block_ct_dim + tile_dst_rt_offset;
        const std::uint32_t l1_tile_idx = base_l1 + block_rt * y_stride + block_c_index * block_ct_dim;
        _llk_pack_untilize_(dest_idx, l1_tile_idx);
    }
}

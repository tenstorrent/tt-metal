// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "llk_param_structs.h"

/*************************************************************************
 * LLK PACK
 *************************************************************************/

/**
 * @brief Initialize packer to pack out a single tile
 *
 * @tparam PACK_SEL: Selects which unpacker resource to use, values = p_pacr::PACK0/PACK1
 * @param pack_output The output circular buffer
 *
 * This function initializes the selected packer to pack a single tile from the destination register to the output
 * circular buffer.
 */
template <std::uint8_t PACK_SEL>
inline void llk_pack_init(const std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);

    _llk_pack_init_<PACK_SEL>(output_id, 1 /*num_tiles_per_pack*/);
}

/**
 *
 * @brief Gets the output L1 tile index where the tile will be packed out to, determined by out_of_order_output
 *
 * @tparam out_of_order_output: Set true to write the output tile to the tile index specified by
 * the user in `output_tile_index`, set false for pack to operqate sequentially: write to the next tile index
 * starting from index 0, and ignore the `output_tile_index` parameter
 * @tparam untilize: Selects pack or pack untilizem
 * @param output_id The output circular buffer identifier
 * @param output_tile_index: The index in the output CB to write to
 *
 * This function packs tiles from the destination register to the output circular buffer.
 *
 */
template <bool out_of_order_output, bool untilize>
inline std::uint32_t get_output_tile_index(std::uint8_t output_id, std::uint32_t output_tile_index) {
    std::uint32_t l1_tile_index;
    if constexpr (out_of_order_output) {
        // Use the write tile index to track position within CB
        l1_tile_index = get_local_cb_interface(output_id).fifo_wr_tile_idx + output_tile_index;
    } else {
        if constexpr (untilize) {
            // TODO: uplift this option from BBE
        } else {
            // In-order packing: use fifo_wr_tile_ptr as the incrementing tile offset
            l1_tile_index = get_local_cb_interface(output_id).fifo_wr_tile_idx +
                            get_local_cb_interface(output_id).fifo_wr_tile_ptr;
            get_local_cb_interface(output_id).fifo_wr_tile_ptr++;
        }
    }
    return l1_tile_index;
}

/**
 *
 * @brief Packs tiles from the destination register to L1 memory
 *
 * @tparam out_of_order_output: Set true to write the output tile to the tile index specified by
 * the user in `output_tile_index`, set false for pack to operqate sequentially: write to the next tile index
 * starting from index 0, and ignore the `output_tile_index` parameter
 * @tparam PACK_SEL: Selects which packer resource to use, values = p_pacr::PACK0/PACK1
 * @param tile_idx: The tile index into the math destination register from where the packer can start packing from
 * @param pack_output The output circular buffer
 * @param output_tile_index: The index in the output CB to write to
 *
 * This function packs tiles from the destination register to the output circular buffer.
 */
template <bool out_of_order_output = false, std::uint8_t PACK_SEL>
inline void llk_pack(
    const std::uint32_t tile_index, const std::uint32_t pack_output, const std::uint32_t output_tile_index = 0) {
    const std::uint8_t output_id = get_output_id(output);
    const std::uint32_t l1_tile_index = get_output_tile_index<out_of_order_output, false>(output_id, output_tile_index);

    _llk_pack_<PACK_SEL>(tile_index, l1_tile_index);
}

/*************************************************************************
 * LLK PACK COMMON
 *************************************************************************/

/**
 * @brief Programs selected packer l1 info & math destination register format
 *
 * @tparam PACK_SEL: Sets which packer to configure, values = p_pacr::PACK0/PACK1
 * @param pack_output The output circular buffer
 */
template <std::uint32_t PACK_SEL>
inline void llk_pack_hw_configure(const std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t base_addr =
        get_local_cb_interface(output_id).fifo_limit - get_local_cb_interface(output_id).fifo_size;

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B = base_addr - 1;
    bd_val.f.format = static_cast<std::uint8_t>(pack_dst_format[output_id]);
    bd_val.f.x_dim       = get_output_face_r_dim(output_id);
    bd_val.f.y_dim = FACE_C_DIM;
    bd_val.f.z_dim       = get_output_num_faces(output_id);

    tdma_descriptor_t td_val;

    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = output_id;
    td_val.reg_data_format = static_cast<std::uint8_t>(pack_src_format[output_id]);

    // TODO: Expand programmability in order to support the dest dvalid scheme with different clients
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
    _llk_pack_hw_configure_<PACK_SEL>(td_val);
}

/**
 * @brief Clears the data valid for destination register after Packer 0 is done packing
 * and zeroes out the dest bank(s) used by packer 0
 *
 * @tparam DST: Destination register buffering mode, values = [DstSync::SyncHalf, DstSync::SyncFull]
 * @tparam IS_FP32_MATH_DEST_EN: flag to show if math destination register is set to float32 mode
 **/
template <DstSync DST, bool IS_FP32_MATH_DEST_EN>
inline void llk_pack_dest_dvalid_section_done() {
    _llk_pack_dest_dvalid_section_done_<DST, IS_FP32_MATH_DEST_EN>();
}

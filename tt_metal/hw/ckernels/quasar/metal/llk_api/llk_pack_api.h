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

/*************************************************************************
 * LLK PACK
 *************************************************************************/

/**
 * @brief Initialize packer to pack out a single tile
 *
 * @param pack_output The output circular buffer
 *
 * This function initializes packer0 to pack a single tile from the destination register to the output
 * circular buffer.
 */
inline void llk_pack_init(const std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);

    _llk_pack_init_<p_pacr::PACK0>(output_id, 1 /*num_tiles_per_pack*/);
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
            l1_tile_index =
                get_local_cb_interface(output_id).fifo_wr_tile_idx + get_local_cb_interface(output_id).fifo_wr_tile_ptr;
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
 * @param tile_idx: The tile index into the math destination register from where the packer can start packing from
 * @param pack_output The output circular buffer
 * @param output_tile_index: The index in the output CB to write to
 *
 * This function packs tiles from the destination register to the output circular buffer, packer0 is used.
 */
template <bool out_of_order_output = false>
inline void llk_pack(
    const std::uint32_t tile_index, const std::uint32_t pack_output, const std::uint32_t output_tile_index = 0) {
    const std::uint8_t output_id = get_output_id(pack_output);
    const std::uint32_t l1_tile_index = get_output_tile_index<out_of_order_output, false>(output_id, output_tile_index);

    _llk_pack_<p_pacr::PACK0>(tile_index, l1_tile_index);
}

/*************************************************************************
 * LLK PACK COMMON
 *************************************************************************/

/**
 * @brief Programs packer0 l1 info & math destination register format
 *
 * @param pack_output The output circular buffer
 */
inline void llk_pack_hw_configure(const std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);

    // Program buffer descriptors for all 32 circular buffers
    for (std::uint32_t i = 0; i < NUM_CIRCULAR_BUFFERS; ++i) {
        const DataFormat l1_data_format = static_cast<DataFormat>(pack_dst_format[i]);

        if (l1_data_format == DataFormat::Invalid) {
            continue;
        }

        buffer_descriptor_u bd_val = {0};
        bd_val.f.l1_addr_16B = get_local_cb_interface(i).fifo_limit - get_local_cb_interface(i).fifo_size;
        bd_val.f.format = static_cast<std::uint8_t>(l1_data_format);
        bd_val.f.x_dim = pack_tile_face_r_dim[i];
        bd_val.f.y_dim = ckernel::trisc::FACE_C_DIM;
        bd_val.f.z_dim = pack_tile_num_faces[i];

        ckernel::trisc::_configure_buf_desc_table_(i, bd_val);
    }

    tdma_descriptor_t td_val;
    td_val.reg_data_format = static_cast<std::uint8_t>(pack_src_format[output_id]);

    // TODO: Expand programmability in order to support the dest dvalid scheme with different clients
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
    _llk_pack_hw_configure_<p_pacr::PACK0>(td_val);
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

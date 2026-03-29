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
#include "experimental/dataflow_buffer.h"

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

    _llk_pack_init_<p_pacr::PACK0>(output_id);
}

/**
 *
 * @brief Gets the output L1 tile index where the tile will be packed out to, determined by out_of_order_output
 *
 * @tparam out_of_order_output: Set true to write the output tile to the tile index specified by
 * the user in `output_tile_index`, set false for pack to operate sequentially: write to the next tile index
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
    LocalDFBInterface& local_dfb_interface = g_dfb_interface[output_id];
    if constexpr (out_of_order_output) {
        // Use the write tile index to track position within DFB
        l1_tile_index = local_dfb_interface.wr_entry_idx + output_tile_index;
    } else {
        if constexpr (untilize) {
            // TODO: uplift this option from BBE
        } else {
            // In-order packing: use fifo_wr_tile_ptr as the incrementing tile offset
            l1_tile_index = local_dfb_interface.wr_entry_idx + local_dfb_interface.wr_entry_ptr;
            local_dfb_interface.wr_entry_ptr++;
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

/**
 * @brief Packs a block of destination tiles into the specified output buffer
 *
 * @param start_tile_index Starting destination register tile index to pack out from
 * @param pack_output Logical output dataflow buffer id
 * @param ntiles Number of consecutive tiles to pack
 *
 * Packs ntiles tiles starting at start_tile_index from the destination register into the L1
 * output buffer identified by pack_output starting from output_tile_index
 */
// TODO: AM; Optimize block calls by using ntiles per pack, issue #40798
inline void llk_pack_block(std::uint32_t start_tile_index, std::uint32_t pack_output, uint32_t ntiles) {
    std::uint8_t output_id = get_output_id(pack_output);

    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        std::uint32_t l1_tile_index = get_output_tile_index<false /* out_of_order_output */, false /* untilize */>(
            output_id, 0 /* output_tile_index */);

        _llk_pack_<p_pacr::PACK0>(tile_index, l1_tile_index);
    }
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

    // Program buffer descriptors for all 32 dataflow buffers, i is the logical dfb id
    for (std::uint32_t i = 0; i < NUM_CIRCULAR_BUFFERS; ++i) {
        const DataFormat l1_data_format = static_cast<DataFormat>(pack_dst_format[i]);

        if (l1_data_format == DataFormat::Invalid) {
            continue;
        }

        // TODO: with multiple TCs are there multiple descriptors?
        buffer_descriptor_u bd_val = {0};
        bd_val.f.l1_addr_16B = g_dfb_interface[i].tc_slots[0].base_addr;
        bd_val.f.format = static_cast<std::uint8_t>(l1_data_format);
        bd_val.f.x_dim = pack_tile_face_r_dim[i];
        bd_val.f.y_dim = ckernel::trisc::FACE_C_DIM;
        bd_val.f.z_dim = pack_tile_num_faces[i];

        ckernel::trisc::_configure_buf_desc_table_(i, bd_val);
    }

    tdma_descriptor_t td_val;
    td_val.reg_data_format = static_cast<std::uint8_t>(pack_src_format[output_id]);

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

/**
 * All the following functions are added to enable Math <-> Pack synchronization
 * on destination register using semaphores.
 *
 * The following functions should be phased out once the dest dvalid scheme is introduced
 */
// TODO: AM; move from semaphores to a per op programmable dest dvalid scheme, issue #37468

/**
 * @brief Waits until math has finished producing data for the current Destination Register section.
 * Blocks on the math–pack semaphore so the packer does not read dest before math has written it.
 */
inline void llk_packer_wait_for_math_done() { _llk_packer_wait_for_math_done_(); }

/**
 * @brief Signals that the packer has finished consuming the current Destination Register section.
 * Posts to the math–pack semaphore and clears/zeros the dest bank(s) used by the packer;
 *
 * @tparam is_fp32_dest_acc_en True if math destination is in 32-bit mode, false for 16-bit mode.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_pack_dest_section_done() {
    _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, DST_SYNC_MODE, is_fp32_dest_acc_en>();
}

/**
 * @brief Configure packer ReLU at runtime from a packed uint32.
 * @param config Packed uint32: bits [1:0] = ReluType, bits [31:16] = threshold.
 */
TT_ALWAYS_INLINE void llk_pack_relu_config(const std::uint32_t config) {
    _llk_pack_relu_config_<p_pacr::PACK0, false>(ckernel::ReluConfig::from_packed(config));
}

TT_ALWAYS_INLINE void llk_pack_relu_config(const ckernel::ReluConfig& relu_config) {
    _llk_pack_relu_config_<p_pacr::PACK0, false>(relu_config);
}

/*************************************************************************
 * LLK PACK REDUCE MASK CONFIGURATION
 *************************************************************************/

/**
 *
 * @brief Configures PACKER0 edge mask programming to support reduce operations
 *
 * @tparam reduce_dim: The reduce op dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 *
 * This function configures the packer edge masks based on the reduce dimension:
 * - REDUCE_ROW: Preserves only datum[0] in each row, masks datums[1:15] to 0 (keeps first column)
 * - REDUCE_COL: Preserves all datums in row 0 only, masks all other rows to 0 (keeps first row)
 * - REDUCE_SCALAR: Preserves only datum[0] in row 0 of face 0 (keeps single element)
 *
 **/
template <ReduceDim reduce_dim>
inline void llk_pack_reduce_mask_config() {
    _llk_pack_reduce_mask_config_<reduce_dim>();
}

/**
 *
 * @brief Clears PACKER0 edge mask configuration to restore normal packing behavior after reduce operations
 *
 * This function disables the edge mask programming for PACKER0 by resetting all masks
 * to preserve all datums in all faces. Should be called after reduce operations to restore
 * normal packing behavior.
 *
 **/
inline void llk_pack_reduce_mask_clear() { _llk_pack_reduce_mask_clear_(); }

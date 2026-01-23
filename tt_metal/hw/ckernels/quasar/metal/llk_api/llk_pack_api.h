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
 * LLK PACK UNTILIZE
 *************************************************************************/

/**
 *
 * @brief Initializes the packer to pack untilize a tile row by full 32x32 tiles
 *
 * @param pack_output: The output circular buffer identifier
 * @param full_ct_dim: Width of a full input in tiles.
 * @param block_ct_dim: Width of a single block in tiles
 *
 * This function initializes pack untilize for a tile row by full 32x32 tiles,
 * from the math destination register to the output circular buffer.
 *
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t BLOCK_CT_DIM>
inline void llk_pack_untilize_init(const std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const TileShape output_tile_shape = {
        .num_faces = get_output_num_faces(output_id),
        .face_r_dim = get_output_face_r_dim(output_id),
        .face_c_dim = FACE_C_DIM,
        .narrow_tile = get_output_narrow_tile(output_id)};

    const std::uint32_t c_dim_faces = (output_tile_shape.narrow_tile ? 1 : 2);
    if (c_dim_faces == 2) {
        _llk_pack_untilize_init_<FULL_CT_DIM, BLOCK_CT_DIM, 2 /*c_dim_faces*/>(output_id, output_tile_shape);
    } else {
        _llk_pack_untilize_init_<FULL_CT_DIM, BLOCK_CT_DIM, 1 /*c_dim_faces*/>(output_id, output_tile_shape);
    }
}

/**
 *
 * @brief Performs pack untilize on a tile row by full 32x32 tiles
 *
 * @tparam block_ct_dim: Width of a single block in tiles
 * @tparam full_ct_dim: Width of a full input in tiles
 * @tparam tile_dst_ct_offset: Compile time offset for the index of the tile in the dest from which to pack
 * @param pack_output: The output circular buffer identifier
 * @param block_c_index: Block column index (used when full_ct_dim > block_ct_dim)
 * @param tile_dst_rt_offset: Runtime offset for the index of the tile in the dest from which to pack
 *
 * This function packs and untilizes a tile row by full 32x32 tiles, from the math destination register to the output
 * circular buffer.
 *
 */
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    std::uint32_t tile_dst_ct_offset = 0>
inline void llk_pack_untilize(
    std::uint32_t block_rt_dim,
    const std::uint32_t pack_output,
    const std::uint32_t block_c_index = 0,
    const std::uint32_t tile_dst_rt_offset = 0) {
    const std::uint32_t output_id = get_output_id(pack_output);

    const TileShape output_tile_shape = {
        .num_faces = get_output_num_faces(output_id),
        .face_r_dim = get_output_face_r_dim(output_id),
        .face_c_dim = ckernel::trisc::FACE_C_DIM,
        .narrow_tile = get_output_narrow_tile(output_id)};
    const std::uint32_t c_dim_faces = (output_tile_shape.narrow_tile ? 1 : 2);  // Tile width in faces
    const std::uint32_t r_dim_faces =
        (output_tile_shape.num_faces == 2 && !output_tile_shape.narrow_tile) ? 1 : 2;  // Tile height in faces

    const std::uint32_t base_l1_index = get_local_cb_interface(output_id).fifo_wr_tile_idx;

    // The internal parts of the strides are applied inside of the _llk_ itself, the external parts are passed to the
    // _llk_pack_untilize_ call x_stride = x_stride_internal = col dim of a tile in L1 in units of 16 datums (1 face);
    // y_stride = y_stride_external + x_stride_internal
    // l1_index = base_l1_index + y * y_stride + x * x_stride;
    const std::uint32_t x = block_c_index * block_ct_dim;
    const std::uint32_t y_stride_external = full_ct_dim * r_dim_faces * output_tile_shape.face_r_dim;

    for (std::uint32_t block_rt = 0; block_rt < block_rt_dim; block_rt++) {
        _llk_pack_untilize_<>(
            block_rt * block_ct_dim + tile_dst_rt_offset + tile_dst_ct_offset,  // dest reg index
            base_l1_index + block_rt /*y*/ * y_stride_external + x)             // l1 index
    }
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

    tdma_descriptor_t td_val;
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

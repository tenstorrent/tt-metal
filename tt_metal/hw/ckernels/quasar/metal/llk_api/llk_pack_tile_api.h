// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"
#include "llk_pack.h"
#include "tensor_shape.h"

/*************************************************************************
 * LLK PACK (tile / block)
 *************************************************************************/

/**
 * @brief Initialize packer to pack out a single tile
 *
 * @param pack_output The output DataFlow Buffer identifier
 *
 * This function initializes packer0 to pack a single tile from the destination register to the output
 * DataFlow Buffer.
 */
inline void llk_pack_init(const std::uint32_t pack_output) {
    const std::uint8_t output_id = static_cast<std::uint8_t>(get_output_id(pack_output));
    const ckernel::TensorShape tensor_shape = get_output_tensor_shape(output_id);

    _llk_pack_init_(output_id, tensor_shape);
}

/**
 *
 * @brief Gets the output L1 tile index where the tile will be packed out to, determined by out_of_order_output
 *
 * @tparam out_of_order_output: Set true to write the output tile to the tile index specified by
 * the user in `output_tile_index`, set false for pack to operate sequentially: write to the next tile index
 * starting from index 0, and ignore the `output_tile_index` parameter
 * @tparam untilize: Selects pack or pack untilizem
 * @param output_id The output DataFlow Buffer identifier
 * @param output_tile_index: The index in the output CB to write to
 *
 * This function packs tiles from the destination register to the output DataFlow Buffer.
 *
 */
template <bool out_of_order_output, bool untilize>
inline std::uint32_t get_output_tile_index(std::uint8_t output_id, std::uint32_t output_tile_index) {
    std::uint32_t l1_tile_index;
    LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(output_id);
    if constexpr (out_of_order_output) {
        // Use the write tile index to track position within DFB
        l1_tile_index = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].wr_entry_idx + output_tile_index;
    } else {
        if constexpr (untilize) {
            // TODO: uplift this option from BBE
        } else {
            // In-order packing: use fifo_wr_tile_ptr as the incrementing tile offset
            l1_tile_index = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].wr_entry_idx +
                            local_dfb_interface.wr_entry_ptr;
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
 * @param pack_output The output DataFlow Buffer identifier
 * @param output_tile_index: The index in the output CB to write to
 *
 * This function packs tiles from the destination register to the output DataFlow Buffer, packer0 is used.
 */
template <bool out_of_order_output = false>
inline void llk_pack(
    const std::uint32_t tile_index, const std::uint32_t pack_output, const std::uint32_t output_tile_index = 0) {
    const std::uint8_t output_id = get_output_id(pack_output);
    const std::uint32_t l1_tile_index = get_output_tile_index<out_of_order_output, false>(output_id, output_tile_index);
    const ckernel::TensorShape tensor_shape = get_output_tensor_shape(output_id);

    _llk_pack_(tile_index, l1_tile_index, tensor_shape);
}

/**
 * @brief Packs a block of destination tiles into the specified output buffer
 *
 * @param start_tile_index Starting destination register tile index to pack out from
 * @param pack_output Logical output DataFlow Buffer identifier
 * @param ntiles Number of consecutive tiles to pack
 *
 * Packs ntiles tiles starting at start_tile_index from the destination register into the L1
 * output buffer identified by pack_output starting from output_tile_index
 */
// TODO: AM; Optimize block calls by using ntiles per pack, issue #40798
inline void llk_pack_block(std::uint32_t start_tile_index, std::uint32_t pack_output, uint32_t ntiles) {
    std::uint8_t output_id = get_output_id(pack_output);
    const ckernel::TensorShape tensor_shape = get_output_tensor_shape(output_id);

    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        std::uint32_t l1_tile_index = get_output_tile_index<false /* out_of_order_output */, false /* untilize */>(
            output_id, 0 /* output_tile_index */);

        _llk_pack_(tile_index, l1_tile_index, tensor_shape);
    }
}

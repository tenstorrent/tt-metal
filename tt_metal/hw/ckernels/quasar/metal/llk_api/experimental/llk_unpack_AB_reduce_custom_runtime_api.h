// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_common_api.h"
#include "experimental/llk_unpack_AB_reduce_runtime_custom.h"

/*************************************************************************
 * LLK UNPACK AB REDUCE CUSTOM (runtime) - Specialized reduce_max_row unpack (Quasar)
 *
 * Mirrors native Quasar reduce (llk_unpack_AB_reduce_init): the input operands are bound at INIT, so the
 * unpack MOP -- which bakes the SrcA buffer descriptor -- is programmed at init. Execute only advances
 * the block-start tile and fires the MOP. No state is carried between init and execute.
 *************************************************************************/

/**
 * @brief Initializes the block reduce_max_row unpacker (runtime block_ct_dim): programs the MOP.
 *
 * Resolves operandA/operandB to buffer descriptors and programs the block MOP.
 *
 * @tparam is_fp32_dest_acc_en  32-bit DEST accumulation mode.
 * @param block_ct_dim     Number of tiles in the width dimension processed as one block.
 * @param respect_trigger  Unsupported on Quasar; must stay false.
 * @param operandA         SrcA operand (block of tiles) circular buffer identifier.
 * @param operandB         SrcB scaler operand circular buffer identifier.
 * @param tensor_shape     Operand tile shape driving the MOP.
 */
template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_AB_reduce_block_max_row_init_runtime(
    const std::uint32_t block_ct_dim,
    const bool respect_trigger,
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const ckernel::TensorShape& tensor_shape) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    _llk_unpack_AB_reduce_block_max_row_init_runtime_<is_fp32_dest_acc_en>(
        block_ct_dim, respect_trigger, operandA_id, operandB_id, tensor_shape);
}

/**
 * @brief Unpacks a block of SrcA operand tiles + one SrcB scaler face (runtime).
 *
 * The block tile count is baked into the MOP by init, so it is not needed here. Resolves operands to L1
 * tile indices, unpacks the scaler once, sets the SrcA block-start tile, and fires the MOP.
 *
 * @param operandA           SrcA operand circular buffer identifier.
 * @param operandB           SrcB scaler operand circular buffer identifier.
 * @param row_start_index    Tile offset of the block's first SrcA tile within operandA's CB.
 * @param tensor_shape       Operand tile shape -- the SAME shape the math thread uses.
 * @param respect_trigger    Unsupported on Quasar; must stay false.
 * @param overlap_first_half Unsupported on Quasar; must stay false.
 */
inline void llk_unpack_AB_reduce_block_max_row_runtime(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t row_start_index,
    const ckernel::TensorShape& tensor_shape,
    const bool respect_trigger = false,
    const bool overlap_first_half = false) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    // Use the caller tensor_shape (single source of truth with the math thread); the CB provides only
    // the physical L1 read pointer below.

    const LocalDFBInterface& local_dfb_interface_a = get_local_dfb_interface(operandA_id);
    const LocalDFBInterface& local_dfb_interface_b = get_local_dfb_interface(operandB_id);
    const std::uint32_t l1_tile_index_a =
        local_dfb_interface_a.tc_slots[local_dfb_interface_a.tc_idx].rd_entry_idx + row_start_index;
    const std::uint32_t l1_tile_index_b = local_dfb_interface_b.tc_slots[local_dfb_interface_b.tc_idx].rd_entry_idx;

    WAYPOINT("URBW");
    // block_ct_dim is baked into the MOP by init; the lib ignores this execute-time argument.
    _llk_unpack_AB_reduce_block_max_row_runtime_(
        0 /*block_ct_dim*/,
        l1_tile_index_a,
        l1_tile_index_b,
        operandB_id,
        tensor_shape,
        respect_trigger,
        overlap_first_half);
    WAYPOINT("URBD");
}

inline void llk_unpack_AB_reduce_block_max_row_uninit_runtime(
    const bool respect_trigger = false, const bool overlap_first_half = false) {
    _llk_unpack_AB_reduce_block_max_row_uninit_runtime_(respect_trigger, overlap_first_half);
}

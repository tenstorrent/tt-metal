// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_common_api.h"
#include "experimental/llk_unpack_AB_reduce_custom.h"

/*************************************************************************
 * LLK UNPACK AB REDUCE CUSTOM - Specialized reduce_max_row unpack (Quasar)
 *
 * Quasar bakes the SrcA buffer descriptor into the unpack MOP (unlike WH/BH, which bind the operand at
 * execute). block_ct_dim is a compile-time template here, so it is available at execute -- the MOP is
 * therefore programmed at EXECUTE (when the operand, hence buffer descriptor, is known), and init only
 * enables the UNPACKER0 transpose. This keeps the init call operand-free and portable with WH/BH.
 *************************************************************************/

/**
 * @brief Initializes the block reduce_max_row unpacker (compile-time block_ct_dim). Enables transpose.
 *
 * @tparam block_ct_dim  Number of tiles in the width dimension processed as one block.
 * @tparam is_fp32_dest_acc_en  32-bit DEST accumulation mode.
 * @tparam respect_trigger  SDPA MOP-split handshake -- unsupported on Quasar; must stay false.
 * @param tensor_shape  Operand tile shape (accepted for WH/BH signature parity; the MOP that uses it is
 *                      programmed at execute).
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false, bool respect_trigger = false>
inline void llk_unpack_AB_reduce_block_max_row_init(const ckernel::TensorShape& tensor_shape) {
    _llk_unpack_AB_reduce_block_max_row_init_<block_ct_dim, is_fp32_dest_acc_en, respect_trigger>(tensor_shape);
}

/**
 * @brief Programs the block reduce_max_row unpack MOP and unpacks a block of SrcA tiles + one scaler.
 *
 * Resolves operandA/operandB to buffer descriptors + L1 tile indices, then programs the MOP for this
 * block (Quasar needs the SrcA buffer descriptor, only known here) and fires it.
 *
 * @tparam block_ct_dim     Number of tiles in the width dimension processed as one block.
 * @tparam respect_trigger  Unsupported on Quasar; must stay false.
 * @param operandA          SrcA operand (block of tiles) circular buffer identifier.
 * @param operandB          SrcB scaler operand circular buffer identifier.
 * @param row_start_index   Tile offset of the block's first SrcA tile within operandA's CB.
 */
template <std::uint32_t block_ct_dim, bool respect_trigger = false>
inline void llk_unpack_AB_reduce_block_max_row(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t row_start_index) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);

    const LocalDFBInterface& local_dfb_interface_a = get_local_dfb_interface(operandA_id);
    const LocalDFBInterface& local_dfb_interface_b = get_local_dfb_interface(operandB_id);
    const std::uint32_t l1_tile_index_a =
        local_dfb_interface_a.tc_slots[local_dfb_interface_a.tc_idx].rd_entry_idx + row_start_index;
    const std::uint32_t l1_tile_index_b = local_dfb_interface_b.tc_slots[local_dfb_interface_b.tc_idx].rd_entry_idx;

    WAYPOINT("URBW");
    _llk_unpack_AB_reduce_block_max_row_<block_ct_dim, respect_trigger>(
        l1_tile_index_a, l1_tile_index_b, operandA_id, operandB_id, tensor_shape);
    WAYPOINT("URBD");
}

template <bool respect_trigger = false>
inline void llk_unpack_AB_reduce_block_max_row_uninit() {
    _llk_unpack_AB_reduce_block_max_row_uninit_<respect_trigger>();
}

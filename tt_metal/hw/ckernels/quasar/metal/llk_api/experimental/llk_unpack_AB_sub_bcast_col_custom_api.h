// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_common_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/llk_unpack_AB_sub_bcast_col_custom.h"

/*************************************************************************
 * LLK UNPACK AB SUB BCAST COL CUSTOM - SDPA blocked bcast-col SUB (Quasar)
 *************************************************************************/

/**
 * @brief Init the unpacker for the SDPA blocked bcast-col SUB path.
 *
 * @param operandA: DFB id of srcA; its tile shape is validated (32x32 or 16x32 tiles).
 * @note Run before @ref llk_unpack_AB_sub_bcast_col_custom on this thread.
 */
inline void llk_unpack_AB_sub_bcast_col_init_custom(const std::uint32_t operandA) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);
    _llk_unpack_AB_sub_bcast_col_init_custom_(tensor_shape);
}

/**
 * @brief SDPA blocked bcast-col unpack: one reused SrcB tile + ct_dim SrcA tiles.
 *
 * operand_id doubles as the buffer-descriptor id (operandA -> UNPACKER0 -> SrcA,
 * operandB -> UNPACKER1 -> SrcB); L1 tile indices are derived from the operand's dataflow buffer.
 *
 * @param operandA: DFB id of srcA (the block of ct_dim column tiles).
 * @param operandB: DFB id of srcB (the single bcast-col tile, reused).
 * @param tile_index_a: First SrcA tile index within operandA.
 * @param tile_index_b: SrcB tile index within operandB.
 * @param ct_dim: Number of SrcA column tiles to unpack.
 * @note Run @ref llk_unpack_AB_sub_bcast_col_init_custom first.
 */
inline void llk_unpack_AB_sub_bcast_col_custom(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t ct_dim = 1) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);

    const LocalDFBInterface& local_dfb_interface_a = get_local_dfb_interface(operandA_id);
    const LocalDFBInterface& local_dfb_interface_b = get_local_dfb_interface(operandB_id);
    const std::uint32_t l1_tile_idx_a =
        local_dfb_interface_a.tc_slots[local_dfb_interface_a.tc_idx].rd_entry_idx + tile_index_a;
    const std::uint32_t l1_tile_idx_b =
        local_dfb_interface_b.tc_slots[local_dfb_interface_b.tc_idx].rd_entry_idx + tile_index_b;

    _llk_unpack_AB_sub_bcast_col_custom_(operandA_id, operandB_id, l1_tile_idx_a, l1_tile_idx_b, ct_dim, tensor_shape);
}

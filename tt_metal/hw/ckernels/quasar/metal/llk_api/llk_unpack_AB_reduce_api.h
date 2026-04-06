// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_reduce.h"

/*************************************************************************
 * LLK UNPACK AB REDUCE
 *************************************************************************/

/**
 *
 * @brief Initialize unpack for unpack reduce operations, which unpacks one tile for srcA and one face for srcB
 *
 * @tparam reduce_dim: Sets the reduce dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * @param operandA: The srcA operand DFB identifier
 * @param operandB: The srcB operand DFB identifier
 *
 * This function initializes the UNPACKER0 to unpack a single tile from the input DFB to srcA
 * and UNPACKER1 to unpack a single face from the input DFB to srcB, with specified reduce dimension.
 *
 */
template <ReduceDim reduce_dim>
inline void llk_unpack_AB_reduce_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);

    _llk_unpack_reduce_init_<reduce_dim>(operandA_id, operandB_id, tensor_shape);
}

/**
 *
 * @brief Unpacks binary operands to SrcA & SrcB for reduce kernels
 *
 * @param operandA: The srcA operand circular buffer identifier
 * @param operandB: The srcB operand circular buffer identifier
 * @param tile_index_a: The L1 index in the input DFB to read from, tile_index_a -> UNPACKER0 -> SRCA
 * @param tile_index_b: The L1 index in the input DFB to read from, tile_index_b -> UNPACKER1 -> SRCB
 *
 * This function performs unpacking for reduce kernels, the UNPACKER0 unpacks a single tile from the input DFB
 * to srcA and UNPACKER1 unpacks a single face from the input DFB to srcB.
 *
 */
inline void llk_unpack_AB_reduce(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    const std::uint32_t l1_tile_index_a = g_dfb_interface[operandA_id].rd_entry_idx + tile_index_a;
    const std::uint32_t l1_tile_index_b = g_dfb_interface[operandB_id].rd_entry_idx + tile_index_b;

    WAYPOINT("UABW");
    _llk_unpack_reduce_(l1_tile_index_a, l1_tile_index_b);
    WAYPOINT("UABD");
}

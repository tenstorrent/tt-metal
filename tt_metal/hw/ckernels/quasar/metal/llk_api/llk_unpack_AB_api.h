// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_binary_operands.h"
#include "llk_unpack_binary_broadcast_operands.h"
#include "llk_unpack_reduce.h"

/*************************************************************************
 * LLK UNPACK AB
 *************************************************************************/

/**
 *
 * @brief Initialize unpack for binary operations, uses SrcA & SrcB, with or without srcB broadcast
 *
 * @tparam src_b_bcast_type: Sets the broadcast type, values = [NONE, COL, ROW, SCALAR]
 * @param operandA: The srcA operand circular buffer identifier
 * @param operandB: The srcB operand circular buffer identifier
 *
 * This function initializes the UNPACKER0 to unpack a single tile from the input circular buffer to srcA
 * and UNPACKER1 to unpack a single tile from the input circular buffer to srcB, with or without srcB broadcast.
 *
 */
template <BroadcastType src_b_bcast_type = BroadcastType::NONE>
inline void llk_unpack_AB_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        _llk_unpack_binary_operands_init_(operandA_id, operandB_id, 1 /*num_tiles_per_unpack*/);
    } else {
        _llk_unpack_binary_broadcast_operands_init_<src_b_bcast_type>(
            operandA_id, operandB_id, 1 /*num_tiles_per_unpack*/);
    }
}

/**
 *
 * @brief Unpacks binary operands to SrcA & SrcB, with or without srcB broadcast
 *
 * @tparam src_b_bcast_type: Sets the broadcast type, values = [NONE, COL, ROW, SCALAR]
 * @param operandA: The srcA operand circular buffer identifier
 * @param operandB: The srcB operand circular buffer identifier
 * @param tile_index_a: The L1 index in the input CB to read from, tile_index_a -> UNPACKER0 -> SRCA
 * @param tile_index_b: The L1 index in the input CB to read from, tile_index_b -> UNPACKER1 -> SRCB
 *
 * This function unpacks binary operands from the input circular buffers to srcA and srcB register,
 * with or without srcB broadcast.
 *
 */
template <BroadcastType src_b_bcast_type = BroadcastType::NONE>
inline void llk_unpack_AB(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);

    std::uint32_t l1_tile_index_a = get_local_cb_interface(operandA_id).fifo_rd_tile_idx + tile_index_a;
    std::uint32_t l1_tile_index_b = get_local_cb_interface(operandB_id).fifo_rd_tile_idx + tile_index_b;

    WAYPOINT("UABW");
    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        _llk_unpack_binary_operands_(l1_tile_index_a, l1_tile_index_b);
    } else {
        _llk_unpack_binary_broadcast_operands_(l1_tile_index_a, l1_tile_index_b);
    }
    WAYPOINT("UABD");
}

/*************************************************************************
 * LLK UNPACK AB REDUCE
 *************************************************************************/

/**
 *
 * @brief Initialize unpack for unpack reduce operations, which unpacks one tile for srcA and one face for srcB
 *
 * @tparam reduce_dim: Sets the reduce dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * @param operandA: The srcA operand circular buffer identifier
 * @param operandB: The srcB operand circular buffer identifier
 *
 * This function initializes the UNPACKER0 to unpack a single tile from the input circular buffer to srcA
 * and UNPACKER1 to unpack a single face from the input circular buffer to srcB, with specified reduce dimension.
 *
 */
template <ReduceDim reduce_dim>
inline void llk_unpack_AB_reduce_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const TileShape tile_shape_A = {
        .num_faces = get_operand_num_faces(operand_id),
        .face_r_dim = get_operand_face_r_dim(operand_id),
        .face_c_dim = FACE_C_DIM,
        .narrow_tile = get_operand_narrow_tile(operand_id)};

    _llk_unpack_reduce_init_<reduce_dim>(operandA_id, operandB_id, 1 /*num_tiles_per_unpack*/, tile_shape_A);
}

/**
 *
 * @brief Unpacks binary operands to SrcA & SrcB for reduce kernels
 *
 * @param operandA: The srcA operand circular buffer identifier
 * @param operandB: The srcB operand circular buffer identifier
 * @param tile_index_a: The L1 index in the input CB to read from, tile_index_a -> UNPACKER0 -> SRCA
 * @param tile_index_b: The L1 index in the input CB to read from, tile_index_b -> UNPACKER1 -> SRCB
 *
 * This function performs unpacking for reduce kernels, the UNPACKER0 unpacks a single tile from the input circular
 * buffer to srcA and UNPACKER1 unpacks a single face from the input circular buffer to srcB.
 *
 */
inline void llk_unpack_AB_reduce(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);

    std::uint32_t l1_tile_index_a = get_local_cb_interface(operandA_id).fifo_rd_tile_idx + tile_index_a;
    std::uint32_t l1_tile_index_b = get_local_cb_interface(operandB_id).fifo_rd_tile_idx + tile_index_b;

    WAYPOINT("UABW");
    _llk_unpack_reduce_(l1_tile_index_a, l1_tile_index_b);
    WAYPOINT("UABD");
}

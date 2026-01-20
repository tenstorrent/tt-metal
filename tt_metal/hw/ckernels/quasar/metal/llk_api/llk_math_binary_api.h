// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_binary_broadcast.h"

/*************************************************************************
 * LLK ELTWISE BINARY
 *************************************************************************/

/**
 *
 * @brief Initialize FPU to perform an elementwise binary operation (if no broadcast) or
 * elementwise binary broadcast operation (if broadcast)
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Sets the broadcast type, values = [NONE, COL, ROW, SCALAR]
 * broadcast only operates on SRCB register
 * @tparam NUM_FIDELITY_PHASES: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication when
 * math is in Tf32 format
 *
 * This function initializes an elementwise binary operation (if no broadcast) or
 * elementwise binary broadcast operation (if broadcast) where:
 * SrcA [+, -, *] SrcB
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * If broadcast type is not NONE, SrcB either has col, row or scalar datums broadcasted
 * to the rest of the tile before elementwise operation
 *
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, std::uint8_t NUM_FIDELITY_PHASES = 0>
inline void llk_math_eltwise_binary_init() {
    const TileShape tile_shape = {.num_faces = 4, .face_r_dim = 16, .face_c_dim = 16, .narrow_tile = false};

    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        _llk_math_eltwise_binary_init_<eltwise_binary_type, NUM_FIDELITY_PHASES, false>(tile_shape);
    } else {
        _llk_math_eltwise_binary_broadcast_init_<eltwise_binary_type, src_b_bcast_type, NUM_FIDELITY_PHASES>(tile_shape);
    }
}

/**
 *
 * @brief Initialize FPU to perform an elementwise binary operation (if no broadcast) or
 * elementwise binary broadcast operation (if broadcast)
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Sets the broadcast type, values = [NONE, COL, ROW, SCALAR]
 * broadcast only operates on SRCB register
 * @tparam NUM_FIDELITY_PHASES: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication when
 * math is in Tf32 format
 * @param operand_A: The srcA input operand circular buffer identifier
 * @param operand_B: The srcB input operand circular buffer identifier
 *
 * This function initializes an elementwise binary operation (if no broadcast) or
 * elementwise binary broadcast operation (if broadcast) where:
 * SrcA [+, -, *] SrcB
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * If broadcast type is not NONE, SrcB either has col, row or scalar datums broadcasted
 * to the rest of the tile before elementwise operation
 *
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, std::uint8_t NUM_FIDELITY_PHASES = 0>
inline void llk_math_eltwise_binary_init_with_operands(const std::uint32_t operand_A, const std::uint32_t operand_B) {
    const std::uint32_t operand_id =
        get_operand_id(operand_A);  // operand_id is used to extract tile dim data which is the same for both operands
    const TileShape tile_shape = {
        .num_faces = get_operand_num_faces(operand_id),
        .face_r_dim = get_operand_face_r_dim(operand_id),
        .face_c_dim = FACE_C_DIM,
        .narrow_tile = get_operand_narrow_tile(operand_id)};

    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        _llk_math_eltwise_binary_init_<eltwise_binary_type, NUM_FIDELITY_PHASES, false>(tile_shape);
    } else {
        _llk_math_eltwise_binary_broadcast_init_<eltwise_binary_type, src_b_bcast_type, NUM_FIDELITY_PHASES>(tile_shape);
    }
}

/**
 *
 * @brief Perform an elementwise binary operation (if no broadcast) or elementwise binary broadcast operation (if
 * broadcast)
 *
 * @tparam src_b_bcast_type: Sets the broadcast type, values = [NONE, COL, ROW, SCALAR]
 * @param dst_index: Tile index into the destination register.
 *
 * This function performs an elementwise binary operation (if no broadcast)
 * or elementwise binary broadcast operation (if broadcast) where:
 * SrcA [+, -, *] SrcB
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * If broadcast type is not NONE, SrcB either has col, row or scalar datums broadcasted
 * to the rest of the tile before elementwise operation
 *
 */
template <BroadcastType src_b_bcast_type>
inline void llk_math_eltwise_binary(std::uint32_t dst_index) {
    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        _llk_math_eltwise_binary_(dst_index);
    } else {
        _llk_math_eltwise_binary_broadcast_(dst_index);
    }
}

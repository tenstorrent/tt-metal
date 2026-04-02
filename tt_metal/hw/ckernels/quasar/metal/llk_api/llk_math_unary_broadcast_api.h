// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_unary_broadcast.h"

/*************************************************************************
 * LLK MATH — unary eltwise with scalar / row / column broadcast (Quasar)
 *************************************************************************/

inline TileShape llk_math_eltwise_unary_broadcast_tile_shape(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    return TileShape{
        .num_faces = get_operand_num_faces(operand_id),
        .face_r_dim = get_operand_face_r_dim(operand_id),
        .face_c_dim = static_cast<std::uint32_t>(ckernel::trisc::FACE_C_DIM),
        .narrow_tile = get_operand_narrow_tile(operand_id) != 0,
    };
}

/**
 * @brief Initialize FPU addrmods / MOP for unary broadcast math (when unpack writes to srcB).
 *
 * @tparam BROADCAST_TYPE Scalar, row, or column broadcast (not NONE).
 * @tparam unpack_to_dest When true, unpack targeted dest; math MOP may be deferred per tt_llk.
 * @tparam is_fp32_dest_acc_en Must be false when unpack_to_dest is true (LLK constraint).
 * @param operand Logical dataflow buffer id for the input operand; tile face geometry is read via
 *                get_operand_num_faces / get_operand_face_r_dim for addrmods.
 */
template <BroadcastType BROADCAST_TYPE, bool unpack_to_dest, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_broadcast_init(const std::uint32_t operand) {
    const TileShape tile_shape = llk_math_eltwise_unary_broadcast_tile_shape(operand);
    _llk_math_eltwise_unary_broadcast_init_<BROADCAST_TYPE, unpack_to_dest, is_fp32_dest_acc_en>(tile_shape);
}

/**
 * @brief Run one destination tile of unary broadcast math (MOVB2D / related MOP).
 *
 * @param dst_index Tile index into the destination register.
 * @param operand Same logical buffer as init; used to rebuild TileShape for each call (required by
 *                _llk_math_eltwise_unary_broadcast_ when unpack_to_dest or template needs it).
 */
template <BroadcastType BROADCAST_TYPE, bool unpack_to_dest, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_broadcast(const std::uint32_t dst_index, const std::uint32_t operand) {
    const TileShape tile_shape = llk_math_eltwise_unary_broadcast_tile_shape(operand);
    _llk_math_eltwise_unary_broadcast_<BROADCAST_TYPE, unpack_to_dest, is_fp32_dest_acc_en>(
        dst_index, tile_shape);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_unary_broadcast.h"

/*************************************************************************
 * LLK MATH — unary eltwise with scalar / row / column broadcast (Quasar)
 * TileShape derived from operand, same idea as llk_math_reduce_init / llk_math_eltwise_unary_datacopy_init.
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
 * @brief Init math for unary broadcast (addrmods / MOP when unpack_to_dest is false).
 */
template <BroadcastType BROADCAST_TYPE, bool unpack_to_dest, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_broadcast_init(const std::uint32_t operand) {
    const TileShape tile_shape = llk_math_eltwise_unary_broadcast_tile_shape(operand);
    _llk_math_eltwise_unary_broadcast_init_<BROADCAST_TYPE, unpack_to_dest, is_fp32_dest_acc_en>(tile_shape);
}

/**
 * @brief Run one output tile of unary broadcast math.
 */
template <BroadcastType BROADCAST_TYPE, bool unpack_to_dest, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_broadcast(const std::uint32_t dst_tile_idx, const std::uint32_t operand) {
    const TileShape tile_shape = llk_math_eltwise_unary_broadcast_tile_shape(operand);
    _llk_math_eltwise_unary_broadcast_<BROADCAST_TYPE, unpack_to_dest, is_fp32_dest_acc_en>(dst_tile_idx, tile_shape);
}

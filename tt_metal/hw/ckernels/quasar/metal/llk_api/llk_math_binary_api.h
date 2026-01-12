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

// Version with no operand
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    int NUM_FIDELITY_PHASES = 0
    bool EN_DI = false>
inline void llk_math_eltwise_binary_init() {
    const TileShape tile_shape = {.num_faces = 4, .face_r_dim = 16, .face_c_dim = 16, .narrow_tile = false};

    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        _llk_math_eltwise_binary_init_<eltwise_binary_type, NUM_FIDELITY_PHASES, EN_DI>(tile_shape);
    } else {
        _llk_math_eltwise_binary_broadcast_init_<eltwise_binary_type, src_b_bcast_type, NUM_FIDELITY_PHASES>(tile_shape);
    }
}

// Version with operands
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    int NUM_FIDELITY_PHASES = 0,
    bool EN_DI = false>
inline void llk_math_eltwise_binary_init_with_operands(
    const std::uint32_t operand_A, const std::uint32_t operand_B) {
    const std::uint32_t operand_id =
        get_operand_id(operand_A);  // operand_id is used to extract tile dim data which is the same for both operands
    const TileShape tile_shape = {.num_faces = get_operand_num_faces(operand_id), .face_r_dim = get_operand_face_r_dim(operand_id), .face_c_dim = 16, .narrow_tile = get_operand_narrow_tile(operand_id)};

    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        _llk_math_eltwise_binary_init_<eltwise_binary_type, NUM_FIDELITY_PHASES, EN_DI>(tile_shape);
    } else {
        _llk_math_eltwise_binary_broadcast_init_<eltwise_binary_type, src_b_bcast_type, NUM_FIDELITY_PHASES>(tile_shape);
    }
}

template <BroadcastType src_b_bcast_type>
inline void llk_math_eltwise_binary(std::uint32_t dst_index) {
    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        _llk_math_eltwise_binary_(dst_index);
    } else {
        _llk_math_eltwise_binary_broadcast_(dst_index);
    }
}

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_eltwise_binary.h"
#include "llk_assert.h"

/*************************************************************************
 * LLK ELTWISE BINARY
 *************************************************************************/

// Version with no operand
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init(const std::uint32_t acc_to_dest = 0) {
    const std::uint32_t num_faces = TILE_NUM_FACES;

    _llk_math_eltwise_binary_init_<eltwise_binary_type, src_b_bcast_type, math_fidelity, binary_reuse_dest>(
        num_faces, acc_to_dest);
}

// Version with operands
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init_with_operands(
    const std::uint32_t operand_A, const std::uint32_t operand_B, const std::uint32_t acc_to_dest = 0) {
    const std::uint32_t operandA_id = get_operand_id(operand_A);
    const std::uint32_t operandB_id = get_operand_id(operand_B);

    // If there is no broadcast, operands must be the same shape. With broadcast, operand B may have different tile
    // dims.
    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        LLK_ASSERT(
            get_operand_num_faces(operandA_id) == get_operand_num_faces(operandB_id),
            "Operands must have same num_faces when src_b_bcast_type == NONE");
        LLK_ASSERT(
            get_operand_face_r_dim(operandA_id) == get_operand_face_r_dim(operandB_id),
            "Operands must have same face_r_dim when src_b_bcast_type == NONE");
    }
    LLK_ASSERT(
        get_operand_src_format(operandA_id) == get_operand_src_format(operandB_id),
        "Operands must have same src format");
    LLK_ASSERT(
        get_operand_dst_format(operandA_id) == get_operand_dst_format(operandB_id),
        "Operands must have same dst format");

    const std::uint32_t num_faces = get_operand_num_faces(operandA_id);

    _llk_math_eltwise_binary_init_<eltwise_binary_type, src_b_bcast_type, math_fidelity, binary_reuse_dest>(
        num_faces, acc_to_dest);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(std::uint32_t dst_index, const bool clear_fp32_dst_acc = true) {
    const std::uint32_t num_faces = TILE_NUM_FACES;

    _llk_math_eltwise_binary_<
        eltwise_binary_type,
        src_b_bcast_type,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        math_fidelity,
        binary_reuse_dest>(num_faces, dst_index, clear_fp32_dst_acc);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(
    const std::uint32_t operand_A,
    const std::uint32_t operand_B,
    std::uint32_t dst_index,
    const bool clear_fp32_dst_acc = true) {
    const std::uint32_t operandA_id = get_operand_id(operand_A);
    const std::uint32_t operandB_id = get_operand_id(operand_B);

    // If there is no broadcast, operands must be the same shape. With broadcast, operand B may have different tile
    // dims.
    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        LLK_ASSERT(
            get_operand_num_faces(operandA_id) == get_operand_num_faces(operandB_id),
            "Operands must have same num_faces when src_b_bcast_type == NONE");
        LLK_ASSERT(
            get_operand_face_r_dim(operandA_id) == get_operand_face_r_dim(operandB_id),
            "Operands must have same face_r_dim when src_b_bcast_type == NONE");
    }
    LLK_ASSERT(
        get_operand_src_format(operandA_id) == get_operand_src_format(operandB_id),
        "Operands must have same src format");
    LLK_ASSERT(
        get_operand_dst_format(operandA_id) == get_operand_dst_format(operandB_id),
        "Operands must have same dst format");

    const std::uint32_t num_faces = get_operand_num_faces(operandA_id);

    _llk_math_eltwise_binary_<
        eltwise_binary_type,
        src_b_bcast_type,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        math_fidelity,
        binary_reuse_dest>(num_faces, dst_index, clear_fp32_dst_acc);
}

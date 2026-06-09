// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB MATMUL
 *************************************************************************/

/**
 * @brief Initialize the unpacker for a matmul (A x B) operation.
 *
 * Programs per-unpacker datum counts (full-tile or face-by-face for partial faces), stashes
 * kt_dim for tile-size scaling, and programs the matmul MOP. Operand A maps to SrcB and
 * operand B to SrcA; partial-face and face/datum geometry are derived from the operands'
 * circular buffers.
 *
 * @param operandA: Circular-buffer index of source operand A (mapped to SrcB).
 * @param operandB: Circular-buffer index of source operand B (mapped to SrcA).
 * @param transpose: Nonzero to enable within-face (16x16) transpose for SrcA.
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param kt_dim: Number of tiles along the contraction (K) dimension.
 * @note Call @ref llk_unpack_AB_matmul with matching template args after this.
 * @ref llk_math_matmul_init is the matching init on the math thread (consumes SrcA/SrcB).
 */
__attribute__((always_inline)) inline void llk_unpack_AB_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // In0 -> srcB (supports partial face)
    // In1 -> srcA
    const uint32_t operandA_id = get_operand_id(operandB);
    const uint32_t operandB_id = get_operand_id(operandA);

    const uint32_t unpA_face_r_dim = get_operand_face_r_dim(operandA_id);
    const uint32_t unpB_face_r_dim = get_operand_face_r_dim(operandB_id);

    const bool reuse_a = ct_dim >= rt_dim;
    const bool partial_face_a = get_operand_partial_face(operandA_id);
    const bool partial_face_b = get_operand_partial_face(operandB_id);

    const uint32_t unpA_num_faces = get_operand_num_faces(operandA_id);
    const uint32_t unpB_num_faces = get_operand_num_faces(operandB_id);  // if partial face -> unpack face by face

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly(
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        unpack_src_format[operandB_id],
        unpack_dst_format[operandB_id],
        unpA_face_r_dim,
        unpB_face_r_dim,
        unpA_num_faces,
        unpB_num_faces));

    _llk_unpack_AB_matmul_init_(
        transpose,
        ct_dim,
        rt_dim,
        kt_dim,
        unpA_face_r_dim,
        unpB_face_r_dim,
        unpA_num_faces,
        unpB_num_faces,
        partial_face_a,
        partial_face_b);
}

/**
 * @brief Unpack the operand tiles for a matmul (A x B) into SrcA and SrcB.
 *
 * Iterates over the reused dimension, computing per-tile L1 addresses from the operands'
 * circular buffers (with kt_dim striding), and unpacks operand A to SrcB / operand B to SrcA
 * for each step while synchronizing through the unpack semaphore.
 *
 * @param operandA: Circular-buffer index of source operand A (mapped to SrcB).
 * @param operandB: Circular-buffer index of source operand B (mapped to SrcA).
 * @param tile_index_a: Starting tile index into operand A's buffer.
 * @param tile_index_b: Starting tile index into operand B's buffer.
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param kt_dim: Number of tiles along the contraction (K) dimension.
 * @note Call @ref llk_unpack_AB_matmul_init before this function.
 * @ref llk_math_matmul on the math thread consumes the SrcA/SrcB tiles unpacked here.
 */
inline void llk_unpack_AB_matmul(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // In0/InA -> srcB (supports partial face)
    // In1/InB -> srcA

    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    // TODO: Review RT, use partial_face_b
    const bool partial_face_a = get_operand_partial_face(operandB_id);
    const bool partial_face_b = get_operand_partial_face(operandA_id);

    std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;

    std::uint32_t tile_size_a = get_local_cb_interface(operandA_id).fifo_page_size;
    std::uint32_t tile_size_b = get_local_cb_interface(operandB_id).fifo_page_size;

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly(
        unpack_src_format[operandB_id],
        unpack_dst_format[operandB_id],
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        get_operand_face_r_dim(operandB_id),
        get_operand_face_r_dim(operandA_id),
        get_operand_num_faces(operandB_id),
        get_operand_num_faces(operandA_id)));

    WAYPOINT("UPMW");
    _llk_unpack_AB_matmul_(
        base_address_a,
        base_address_b,
        tile_index_a,
        tile_index_b,
        tile_size_a,
        tile_size_b,
        partial_face_a,
        partial_face_b,
        ct_dim,
        rt_dim,
        kt_dim);
    WAYPOINT("UPMD");
}

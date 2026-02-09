// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB MATMUL
 *************************************************************************/

__attribute__((always_inline)) inline void llk_unpack_AB_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // In0 -> srcB (supports partial face)
    // In1 -> srcA
    const std::uint32_t operandA_id = get_operand_id(operandB);
    const std::uint32_t operandB_id = get_operand_id(operandA);

    const std::uint32_t unpA_face_r_dim = get_operand_face_r_dim(operandA_id);
    const std::uint32_t unpB_face_r_dim = get_operand_face_r_dim(operandB_id);

    const bool partial_face_a = get_operand_partial_face(operandA_id);
    const bool partial_face_b = get_operand_partial_face(operandB_id);

    const std::uint32_t unpA_num_faces = partial_face_a ? 1 : get_operand_num_faces(operandA_id);
    const std::uint32_t unpB_num_faces =
        partial_face_b ? 1 : get_operand_num_faces(operandB_id);  // if partial face -> unpack face by face

    LLK_ASSERT(ct_dim > 0, "ct_dim must be > 0");
    LLK_ASSERT(rt_dim > 0, "rt_dim must be > 0");
    LLK_ASSERT(kt_dim > 0, "kt_dim must be > 0");

    // Validate operand constraints
    // In0 (SrcB): narrow_tile must be False
    LLK_ASSERT(!get_operand_narrow_tile(operandA_id), "In0/SrcB: narrow_tile must be False");

    // In1 (SrcA): face_r_dim must be 16
    LLK_ASSERT(unpB_face_r_dim == 16, "In1/SrcA: face_r_dim must be 16");

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

    // TODO: remove partial_face flag, as this is easily to be confused with the partial face flag in math kernel
    const bool partial_face_a = get_operand_partial_face(operandB_id);  // In1/InB -> srcA
    const bool partial_face_b = get_operand_partial_face(operandA_id);  // In0/InA -> srcB

    const std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    const std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;
    const std::uint32_t tile_size_a = get_local_cb_interface(operandA_id).fifo_page_size;
    const std::uint32_t tile_size_b = get_local_cb_interface(operandB_id).fifo_page_size;

    LLK_ASSERT(ct_dim > 0, "ct_dim must be > 0");
    LLK_ASSERT(rt_dim > 0, "rt_dim must be > 0");
    LLK_ASSERT(kt_dim > 0, "kt_dim must be > 0");

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

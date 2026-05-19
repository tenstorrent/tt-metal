// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
    const uint32_t operandA_id = get_operand_id(operandB);
    const uint32_t operandB_id = get_operand_id(operandA);

    const uint32_t unpA_face_r_dim = get_operand_face_r_dim(operandA_id);
    const uint32_t unpB_face_r_dim = get_operand_face_r_dim(operandB_id);

    const bool reuse_a = ct_dim >= rt_dim;
    const bool partial_face_a = get_operand_partial_face(operandA_id);
    const bool partial_face_b = get_operand_partial_face(operandB_id);

    const uint32_t unpA_num_faces = partial_face_a ? 1 : get_operand_num_faces(operandA_id);
    const uint32_t unpB_num_faces =
        partial_face_b ? 1 : get_operand_num_faces(operandB_id);  // if partial face -> unpack face by face

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

    // Re-establish TILE_SIZE_A/B GPRs after the LLK init runs. The matmul op
    // reads these GPRs via inline asm in _llk_unpack_AB_matmul_; if any
    // intervening kernel reconfigure (e.g. *_with_dt format swap on another
    // CB) left them holding the wrong values, the matmul addressing breaks.
    // These writes must come AFTER _llk_unpack_AB_matmul_init_ so they are
    // the last touch of these GPRs before the op runs.
    // GPR naming follows the unpacker (unpA <- In0, unpB <- In1).
    const uint32_t unpA_operand_id = get_operand_id(operandA);
    const uint32_t unpB_operand_id = get_operand_id(operandB);
    const uint32_t unpA_tile_size = get_local_cb_interface(unpA_operand_id).fifo_page_size;
    const uint32_t unpB_tile_size = get_local_cb_interface(unpB_operand_id).fifo_page_size;
    TT_SETDMAREG(0, LOWER_HALFWORD(unpA_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    TT_SETDMAREG(0, LOWER_HALFWORD(unpB_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
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
        partial_face_a ? 1 : get_operand_num_faces(operandB_id),
        partial_face_b ? 1 : get_operand_num_faces(operandA_id)));

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

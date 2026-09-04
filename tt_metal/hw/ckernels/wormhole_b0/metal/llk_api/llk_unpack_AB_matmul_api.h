// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common_api.h"
#include "sanitizer/api.h"

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

    llk::san::unpack_operand_check(
        llk::san::IGNORE,
        unpack_src_format[operandA_id],
        unpack_src_format[operandB_id],
        unpack_dst_format[operandA_id],
        unpack_dst_format[operandB_id],
        llk::san::IGNORE,
        llk::san::IGNORE,
        llk::san::IGNORE,
        llk::san::IGNORE);

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

// Program the unpacker's SrcA tile-size register from an explicit tile size instead of operand B's
// circular-buffer page size. The matmul MOP steps SrcA's L1 base by that register across ct_dim,
// independently of the base address llk_unpack_AB_matmul_at is handed, so a buffer paged coarser
// than a tile needs both. (Operand B feeds SrcA in matmul; see the role swap above.) Face geometry
// is restated unchanged from the operand's own circular-buffer metadata.
inline void llk_unpack_AB_matmul_set_operand_b_tile_size(const std::uint32_t operandB, const std::uint32_t tile_size) {
    const std::uint32_t operandB_id = get_operand_id(operandB);
    _llk_unpack_reconfig_tile_shape_srca_(
        tile_size, get_operand_face_r_dim(operandB_id), get_operand_num_faces(operandB_id));
}

// Matmul unpack with operand B's tiles addressed from an explicit read pointer and tile size
// instead of operand B's circular-buffer page stride. A consumer whose operand-B buffer is paged
// coarser than a tile -- one K-block per page, which is what PrefetcherPipe delivery hands the 1D
// matmul -- reads the block's pointer once and indexes the tiles inside it.
//
// read_ptr_b and tile_size_b are in the units LocalCBInterface uses (16-byte words on TRISC); pass
// the read pointer exactly as the interface reports it, the hardware's -1 base offset is applied
// here. operandB is still needed: its data format and face geometry stay circular-buffer derived.
inline void llk_unpack_AB_matmul_at(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t read_ptr_b,
    const std::uint32_t tile_size_b,
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
    const bool partial_face_b = get_operand_partial_face(operandA_id);  // In0/InA -> srcB`

    std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    std::uint32_t base_address_b = read_ptr_b - 1;
    std::uint32_t tile_size_a = get_local_cb_interface(operandA_id).fifo_page_size;

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly(
        unpack_src_format[operandB_id],
        unpack_dst_format[operandB_id],
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        get_operand_face_r_dim(operandB_id),
        get_operand_face_r_dim(operandA_id),
        get_operand_num_faces(operandB_id),
        get_operand_num_faces(operandA_id)));

    llk::san::unpack_operand_check(
        llk::san::IGNORE,
        unpack_src_format[operandB_id],
        unpack_src_format[operandA_id],
        unpack_dst_format[operandB_id],
        unpack_dst_format[operandA_id],
        get_operand_face_r_dim(operandB_id),
        get_operand_face_r_dim(operandA_id),
        partial_face_a ? 1 : get_operand_num_faces(operandB_id),
        partial_face_b ? 1 : get_operand_num_faces(operandA_id));

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

inline void llk_unpack_AB_matmul(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    const std::uint32_t operandB_id = get_operand_id(operandB);
    llk_unpack_AB_matmul_at(
        operandA,
        operandB,
        get_local_cb_interface(operandB_id).fifo_rd_ptr,
        get_local_cb_interface(operandB_id).fifo_page_size,
        tile_index_a,
        tile_index_b,
        ct_dim,
        rt_dim,
        kt_dim);
}

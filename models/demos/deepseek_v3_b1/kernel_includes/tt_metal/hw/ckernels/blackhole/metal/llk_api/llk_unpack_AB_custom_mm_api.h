// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../../../../../third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_custom_mm.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB CUSTOM_MM
 *
 * Custom matmul that uses MOP to loop both srcA and srcB along inner dim. Output height
 * and width should be single tile with tile shape [1, 32]. Further work will uplift the
 * custom mm to support for tiles along the width.
 *
 * Simplified API containing only functions used by custom_mm_block_init and custom_mm_block.
 * Uses llk_unpack_AB_custom_mm.h as the low-level implementation.
 *************************************************************************/

template <bool is_fp32_dest_acc_en, StochRndType stoch_rnd_mode = StochRndType::None>
inline void llk_unpack_AB_custom_mm_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpB_operand, const std::uint32_t transpose_xy_srca = 0) {
    // In0 -> unpB
    // In1 -> unpA
    const uint32_t unpA_operand_id = get_operand_id(unpB_operand);
    const uint32_t unpB_operand_id = get_operand_id(unpA_operand);

    // unpA -> srcA
    // unpB -> srcB
    const uint32_t unpA_num_faces = get_operand_num_faces(unpA_operand_id);
    const uint32_t unpB_num_faces = get_operand_num_faces(unpB_operand_id);

    const uint32_t unpA_face_r_dim = get_operand_face_r_dim(unpA_operand_id);
    const uint32_t unpB_face_r_dim = get_operand_face_r_dim(unpB_operand_id);

    _llk_unpack_AB_custom_mm_hw_configure_<is_fp32_dest_acc_en, stoch_rnd_mode>(
        unpack_src_format[unpA_operand_id],
        unpack_src_format[unpB_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpack_dst_format[unpB_operand_id],
        unpA_face_r_dim,
        unpB_face_r_dim,
        transpose_xy_srca,
        unpA_num_faces,
        unpB_num_faces,
        get_local_cb_interface(unpA_operand_id).fifo_page_size,
        get_local_cb_interface(unpB_operand_id).fifo_page_size);
}

__attribute__((always_inline)) inline void llk_unpack_AB_custom_mm_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t kt_dim = 1) {
    // In0 -> srcB (supports partial face)
    // In1 -> srcA
    const uint32_t operandA_id = get_operand_id(operandB);
    const uint32_t operandB_id = get_operand_id(operandA);

    const uint32_t unpA_face_r_dim = get_operand_face_r_dim(operandA_id);
    const uint32_t unpB_face_r_dim = get_operand_face_r_dim(operandB_id);

    const bool partial_face_b = get_operand_partial_face(operandB_id);

    const uint32_t unpA_num_faces = get_operand_num_faces(operandA_id);
    const uint32_t unpB_num_faces =
        partial_face_b ? 1 : get_operand_num_faces(operandB_id);  // if partial face -> unpack face by face

    _llk_unpack_AB_custom_mm_init_(
        kt_dim, unpA_face_r_dim, unpB_face_r_dim, unpA_num_faces, unpB_num_faces, partial_face_b);
}

inline void llk_unpack_AB_custom_mm(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t kt_dim = 1) {
    // In0/InA -> srcB (supports partial face)
    // In1/InB -> srcA

    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;

    std::uint32_t tile_size_a = get_local_cb_interface(operandA_id).fifo_page_size;
    std::uint32_t tile_size_b = get_local_cb_interface(operandB_id).fifo_page_size;

    WAYPOINT("UPMW");
    _llk_unpack_AB_custom_mm_(
        base_address_a, base_address_b, tile_index_a, tile_index_b, tile_size_a, tile_size_b, kt_dim);
    WAYPOINT("UPMD");
}

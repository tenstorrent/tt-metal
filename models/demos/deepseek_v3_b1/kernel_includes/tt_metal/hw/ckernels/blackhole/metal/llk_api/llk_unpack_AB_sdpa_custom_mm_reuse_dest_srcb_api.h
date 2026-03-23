// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../../../../../third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB SDPA_CUSTOM_MM_REUSE_DEST_SRCB
 *
 * Custom matmul that reuses SrcB from dest and only unpacks SrcA. Output height
 * and width should be single tile with tile shape [1, 32]. Further work will uplift the
 * custom mm to support for tiles along the width.
 *
 * Uses generic llk_unpack_hw_configure for hardware configuration (operation
 * specific hw_configures are deprecated). Uses
 * llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb.h as the low-level implementation.
 *************************************************************************/

__attribute__((always_inline)) inline void llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init(
    const std::uint32_t operand0,
    const std::uint32_t operand1,
    const std::uint32_t transpose = 0,
    const std::uint32_t nt_dim = 1) {
    const std::uint32_t operandA_id = get_operand_id(operand1);

    const uint32_t unpA_face_r_dim = get_operand_face_r_dim(operandA_id);
    const uint32_t unpA_num_faces = get_operand_num_faces(operandA_id);

    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init_(nt_dim, unpA_face_r_dim, unpA_num_faces);
}

inline void llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb(
    const std::uint32_t operand0,
    const std::uint32_t operand1,
    const std::uint32_t tile_index_0,
    const std::uint32_t tile_index_1,
    const std::uint32_t kt_dim = 1,
    const std::uint32_t nt_dim = 1,
    const std::uint32_t in1_k_stride = 1) {
    const std::uint32_t operandA_id = get_operand_id(operand1);

    const std::uint32_t base_address_A = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    const std::uint32_t tile_index_A = tile_index_1;
    const std::uint32_t tile_size_A = get_local_cb_interface(operandA_id).fifo_page_size;

    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_(
        base_address_A, tile_index_A, tile_size_A, kt_dim, nt_dim, in1_k_stride);
}

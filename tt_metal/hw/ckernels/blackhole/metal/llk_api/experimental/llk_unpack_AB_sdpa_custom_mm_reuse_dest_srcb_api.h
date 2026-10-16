// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "experimental/llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb.h"
#include "llk_unpack_common_api.h"
#include "sanitizer/api.h"

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

    const std::uint32_t unpA_face_r_dim = get_operand_face_r_dim(operandA_id);
    const std::uint32_t unpA_num_faces = get_operand_num_faces(operandA_id);

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
    SAN_HOOK(unsupported());
    const std::uint32_t operandA_id = get_operand_id(operand1);

    const std::uint32_t base_address_A = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    const std::uint32_t tile_index_A = tile_index_1;
    const std::uint32_t tile_size_A = get_local_cb_interface(operandA_id).fifo_page_size;

    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_(
        base_address_A, tile_index_A, tile_size_A, kt_dim, nt_dim, in1_k_stride);
}

/**
 * @brief Unpack one OV K row for the SrcB-reuse matmul.
 *
 * @param operand: Circular buffer containing the V tiles.
 * @param tile_index: Index of the first V tile relative to the buffer read pointer.
 * @param nt_dim: Number of contiguous V tiles in the row, from 1 to 16.
 * @note Call @ref llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init with the same nt_dim first;
 *       do not reload the replay buffer before this call. Call @ref llk_unpack_A_sdpa_set_srcb_dummy_valid
 *       before unpacking and pair with @ref llk_math_sdpa_custom_mm_reuse_dest_srcb at kt_dim = 1.
 */
inline void llk_unpack_A_sdpa_custom_mm_single_row(
    const std::uint32_t operand, const std::uint32_t tile_index, const std::uint32_t nt_dim) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t tile_size = get_local_cb_interface(operand_id).fifo_page_size;
    const std::uint32_t address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1 + tile_index * tile_size;
    _llk_unpack_A_sdpa_custom_mm_single_row_(address, tile_size, nt_dim);
}

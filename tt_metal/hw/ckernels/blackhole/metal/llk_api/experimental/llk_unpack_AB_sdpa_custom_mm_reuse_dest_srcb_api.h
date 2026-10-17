// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <type_traits>
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
    const std::uint32_t operand0, const std::uint32_t operand1, const std::uint32_t nt_dim) {
    // An integer at the former transpose position binds here as nt_dim = 0 or 1: 0 is caught here,
    // 1 by the kt_dim assert of llk_math_sdpa_custom_mm_reuse_dest_srcb_init (kt_dim binds as 1).
    LLK_ASSERT(nt_dim >= 1, "llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init: nt_dim must be at least 1");
    const std::uint32_t operandA_id = get_operand_id(operand1);

    const std::uint32_t unpA_face_r_dim = get_operand_face_r_dim(operandA_id);
    const std::uint32_t unpA_num_faces = get_operand_num_faces(operandA_id);

    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init_(nt_dim, unpA_face_r_dim, unpA_num_faces);
}

// Legacy (operand0, operand1, transpose) argument lists have the same arity as the current
// (operand0, operand1, nt_dim) list. This overload is viable only for an exact bool at the old
// transpose position and is deleted, so a bool transpose does not re-bind as nt_dim; integers
// resolve to the live overload and its nt_dim assert. A transpose of operand B is not
// implemented: see the transpose_v static_assert in compute_sdpa_chunk.
template <typename T, std::enable_if_t<std::is_same_v<T, bool>, int> = 0>
inline void llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init(
    const std::uint32_t operand0, const std::uint32_t operand1, T transpose) = delete;

inline void llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb(
    const std::uint32_t operand0,
    const std::uint32_t operand1,
    const std::uint32_t tile_index_0,
    const std::uint32_t tile_index_1,
    const std::uint32_t kt_dim,
    const std::uint32_t nt_dim,
    const std::uint32_t in1_k_stride) {
    SAN_HOOK(unsupported());
    // The K loop issues a MOP for every pair of K tiles, kt_dim / 2 - 1 times after the first:
    // kt_dim below 2 underflows that count, and an odd kt_dim feeds only 2 * (kt_dim / 2) K tiles
    // while the math side loops kt_dim times. An integer at the former transpose position of the
    // compute API binds as kt_dim = 0 or 1 and is caught here.
    LLK_ASSERT(
        kt_dim >= 2 && (kt_dim % 2) == 0 && nt_dim >= 1,
        "llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb: kt_dim must be even and at least 2, nt_dim at least 1");
    const std::uint32_t operandA_id = get_operand_id(operand1);

    const std::uint32_t base_address_A = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    const std::uint32_t tile_index_A = tile_index_1;
    const std::uint32_t tile_size_A = get_local_cb_interface(operandA_id).fifo_page_size;

    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_(
        base_address_A, tile_index_A, tile_size_A, kt_dim, nt_dim, in1_k_stride);
}

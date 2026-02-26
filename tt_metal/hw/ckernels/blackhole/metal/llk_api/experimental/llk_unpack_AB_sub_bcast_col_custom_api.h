// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_common_api.h"
#include "experimental/llk_unpack_AB_sub_bcast_col_custom.h"

/*************************************************************************
 * LLK UNPACK AB SUB BCAST COL CUSTOM - SDPA specialized blocked sub path
 *************************************************************************/

template <BroadcastType BType = BroadcastType::COL>
inline void llk_unpack_AB_sub_bcast_col_init_custom(const std::uint32_t operandA) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operandA_id);
    const std::uint32_t num_faces = get_operand_num_faces(operandA_id);
    const bool narrow_tile = get_operand_narrow_tile(operandA_id);

    _llk_unpack_AB_sub_bcast_col_init_custom_<BType>(face_r_dim, num_faces, narrow_tile);
}

template <BroadcastType BType = BroadcastType::COL>
inline void llk_unpack_AB_sub_bcast_col_custom(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t ct_dim = 1) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    const std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    const std::uint32_t offset_address_a = get_local_cb_interface(operandA_id).fifo_page_size * tile_index_a;
    const std::uint32_t address_a = base_address_a + offset_address_a;

    const std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;
    const std::uint32_t offset_address_b = get_local_cb_interface(operandB_id).fifo_page_size * tile_index_b;
    const std::uint32_t address_b = base_address_b + offset_address_b;

    _llk_unpack_AB_sub_bcast_col_custom_<BType>(address_a, address_b, ct_dim);
}

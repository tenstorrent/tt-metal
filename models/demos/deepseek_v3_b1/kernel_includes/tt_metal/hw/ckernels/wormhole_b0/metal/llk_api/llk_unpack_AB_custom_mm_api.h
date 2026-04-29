// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../../../../../third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_AB_custom_mm.h"
#include "llk_unpack_common_api.h"

template <bool transpose = false>
inline void llk_unpack_AB_custom_mm_init(
    const std::uint32_t operand0, const std::uint32_t operand1, const std::uint32_t ct_dim = 1) {
    const std::uint32_t operandA_id = get_operand_id(operand1);
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);
    const std::uint32_t operandA_unpack_dst_format = unpack_dst_format[operandA_id];

    _llk_unpack_AB_custom_mm_init_<transpose>(operandB_face_r_dim, operandA_unpack_dst_format, ct_dim);
}

template <bool read_transposed = false, bool clear_src = true>
inline void llk_unpack_AB_custom_mm(
    const std::uint32_t operand0,
    const std::uint32_t operand1,
    const std::uint32_t tile_index_0,
    const std::uint32_t tile_index_1,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    const std::uint32_t operandA_id = get_operand_id(operand1);
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t base_address_A = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    const std::uint32_t base_address_B = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;
    const std::uint32_t tile_index_A = tile_index_1;
    const std::uint32_t tile_index_B = tile_index_0;
    const std::uint32_t tile_size_A = get_local_cb_interface(operandA_id).fifo_page_size;
    const std::uint32_t tile_size_B = get_local_cb_interface(operandB_id).fifo_page_size;

    _llk_unpack_AB_custom_mm_<read_transposed, clear_src>(
        base_address_A, base_address_B, tile_index_A, tile_index_B, tile_size_A, tile_size_B, kt_dim, ct_dim);
}

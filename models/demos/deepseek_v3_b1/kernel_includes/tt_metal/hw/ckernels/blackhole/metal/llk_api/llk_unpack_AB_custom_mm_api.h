// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../../../../../third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_custom_mm.h"
#include "llk_unpack_common_api.h"

template <bool transpose = false>
inline void llk_unpack_AB_custom_mm_init(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t ct_dim = 1) {
    const std::uint32_t operandB_id = get_operand_id(operandA);
    const std::uint32_t unpB_face_r_dim = get_operand_face_r_dim(operandB_id);

    _llk_unpack_AB_custom_mm_init_<transpose>(unpB_face_r_dim, ct_dim);
}

template <bool read_transposed = false>
inline void llk_unpack_AB_custom_mm(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    const std::uint32_t operandA_id = get_operand_id(operandB);
    const std::uint32_t operandB_id = get_operand_id(operandA);
    std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;
    std::uint32_t tile_size_a = get_local_cb_interface(operandA_id).fifo_page_size;
    std::uint32_t tile_size_b = get_local_cb_interface(operandB_id).fifo_page_size;

    _llk_unpack_AB_custom_mm_<read_transposed>(
        base_address_a, base_address_b, tile_index_b, tile_index_a, tile_size_a, tile_size_b, kt_dim, ct_dim);
}

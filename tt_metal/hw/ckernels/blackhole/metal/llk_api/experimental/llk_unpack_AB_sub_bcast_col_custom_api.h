// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_unpack_common_api.h"
#include "experimental/llk_unpack_AB_sub_bcast_col_custom.h"

/*************************************************************************
 * LLK UNPACK AB SUB BCAST COL CUSTOM - SDPA specialized blocked sub path
 *************************************************************************/

inline void llk_unpack_AB_sub_bcast_col_init_custom(const std::uint32_t operandA) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);
    _llk_unpack_AB_sub_bcast_col_init_custom_(tensor_shape);
}

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

    LLK_ASSERT(cb_access_within_bounds(operandA_id, tile_index_a, 1), "Indexed tile read exceeds CB boundary");
    LLK_ASSERT(cb_access_within_bounds(operandB_id, tile_index_b, 1), "Indexed tile read exceeds CB boundary");

    _llk_unpack_AB_sub_bcast_col_custom_(address_a, address_b, ct_dim);
}

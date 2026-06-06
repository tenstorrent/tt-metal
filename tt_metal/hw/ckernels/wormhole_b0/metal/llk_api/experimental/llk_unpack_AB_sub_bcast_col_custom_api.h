// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_unpack_AB_sub_bcast_col_custom.h"
#include "llk_unpack_cb_tile_access.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB SUB BCAST COL CUSTOM - SDPA specialized blocked sub path
 *************************************************************************/

inline void llk_unpack_AB_sub_bcast_col_init_custom(const std::uint32_t /*operandA*/) {
    _llk_unpack_AB_sub_bcast_col_init_custom_();
}

inline void llk_unpack_AB_sub_bcast_col_custom(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t ct_dim = 1) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    const std::uint32_t address_a = llk_unpack_tile_address(operandA_id, tile_index_a);
    const std::uint32_t address_b = llk_unpack_tile_address(operandB_id, tile_index_b);

    LLK_ASSERT_BLOCK(validate_unpack_tile_access(operandA_id, tile_index_a, 1));
    LLK_ASSERT_BLOCK(validate_unpack_tile_access(operandB_id, tile_index_b, 1));

    _llk_unpack_AB_sub_bcast_col_custom_(address_a, address_b, ct_dim);
}

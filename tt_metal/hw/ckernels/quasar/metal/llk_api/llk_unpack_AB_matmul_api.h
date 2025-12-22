// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_matmul.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB MATMUL
 *************************************************************************/

template <bool TRANSPOSE_EN = false>
__attribute__((always_inline)) inline void llk_unpack_AB_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // In0 -> srcB (supports partial face)
    // In1 -> srcA
    const uint32_t operandA_id = get_operand_id(operandA);
    const uint32_t operandB_id = get_operand_id(operandB);

    _llk_unpack_matmul_init_<TRANSPOSE_EN>(operandA_id, operandB_id, ct_dim, rt_dim, kt_dim);
}

inline void llk_unpack_AB_matmul(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // In0/InA -> srcB
    // In1/InB -> srcA

    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    std::uint32_t l1_tile_idx_0 = get_local_cb_interface(operandA_id).fifo_rd_tile_idx + tile_index_a;
    std::uint32_t l1_tile_idx_1 = get_local_cb_interface(operandB_id).fifo_rd_tile_idx + tile_index_b;

    WAYPOINT("UPMW");
    _llk_unpack_matmul_(ct_dim, rt_dim, kt_dim, l1_tile_idx_0, l1_tile_idx_1);
    WAYPOINT("UPMD");
}

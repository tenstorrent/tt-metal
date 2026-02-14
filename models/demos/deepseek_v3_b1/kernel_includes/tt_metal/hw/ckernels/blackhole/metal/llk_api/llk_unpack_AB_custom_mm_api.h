// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../../../../../third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_custom_mm.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB CUSTOM_MM
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: {1, 2, 4, 6, 8, 10, 12, 14, 16}
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Uses llk_unpack_AB_custom_mm.h as the low-level implementation.
 *************************************************************************/

template <bool transpose = false>
inline void llk_unpack_AB_custom_mm_init(
    const std::uint32_t operand0, const std::uint32_t operand1, const std::uint32_t ct_dim = 1) {
    // Swap operands, for matmul operand0 goes to SrcB and operand1 goes to SrcA
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);

    _llk_unpack_AB_custom_mm_init_<transpose>(operandB_face_r_dim, ct_dim);
}

template <bool read_transposed = false>
inline void llk_unpack_AB_custom_mm(
    const std::uint32_t operand0,
    const std::uint32_t operand1,
    const std::uint32_t tile_index_0,
    const std::uint32_t tile_index_1,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    // Swap operands, for matmul operand0 goes to SrcB and operand1 goes to SrcA
    const std::uint32_t operandA_id = get_operand_id(operand1);
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t base_address_A = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    const std::uint32_t base_address_B = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;
    const std::uint32_t tile_index_A = tile_index_1;
    const std::uint32_t tile_index_B = tile_index_0;
    const std::uint32_t tile_size_A = get_local_cb_interface(operandA_id).fifo_page_size;
    const std::uint32_t tile_size_B = get_local_cb_interface(operandB_id).fifo_page_size;

    _llk_unpack_AB_custom_mm_<read_transposed>(
        base_address_A, base_address_B, tile_index_A, tile_index_B, tile_size_A, tile_size_B, kt_dim, ct_dim);
}

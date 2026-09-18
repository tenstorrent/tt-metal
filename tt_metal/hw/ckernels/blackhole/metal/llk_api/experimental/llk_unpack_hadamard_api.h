// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_common_api.h"
#include "experimental/llk_unpack_hadamard.h"

// One-shot unpack init for the H128 transform. Configures the unpackers
// for single-face (16x16) operands and preprograms context 1's srcA base
// with the H_16 tile's L1 address. Call once during init, after
// llk_unpack_hw_configure. H_16 must stay resident at this address for
// the program's lifetime. Operand order: operandA = h16, operandB = input.
inline void llk_unpack_hadamard_h128_init(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t h16_tile_index) {
    const std::uint32_t operandA_id = get_operand_id(operandA);  // h16
    LLK_ASSERT(get_operand_num_faces(operandA_id) == 1, "Hadamard H128 unpack requires single-face h16 operand");
    LLK_ASSERT(
        get_operand_num_faces(get_operand_id(operandB)) == 1,
        "Hadamard H128 unpack requires single-face input operand");

    const std::uint32_t base_address = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    const std::uint32_t address = base_address + get_local_cb_interface(operandA_id).fifo_page_size * h16_tile_index;
    _llk_unpack_hadamard_h128_init_(address);
}

// One tile's worth of Hadamard unpack: phase 1 (h16 -> srcB, input ->
// srcA bank 0, zeroed + narrowed to 8 rows) and phase 2 (H_16 -> srcA
// bank 1). Operand order: operandA = h16 (srcB), operandB = input (srcA).
inline void llk_unpack_hadamard_h128(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b) {
    const std::uint32_t operandA_id = get_operand_id(operandA);  // h16
    const std::uint32_t operandB_id = get_operand_id(operandB);  // input

    const std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    const std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;

    const std::uint32_t tile_size_a = get_local_cb_interface(operandA_id).fifo_page_size;
    const std::uint32_t tile_size_b = get_local_cb_interface(operandB_id).fifo_page_size;

    WAYPOINT("UPHW");
    _llk_unpack_hadamard_h128_(base_address_a, base_address_b, tile_index_a, tile_index_b, tile_size_a, tile_size_b);
    WAYPOINT("UPHD");
}

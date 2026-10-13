// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_common_api.h"
#include "experimental/llk_unpack_hadamard.h"
#include "sanitizer/api.h"

// One-shot unpack init for the H128 transform. Configures the unpackers
// for single-face (16x16) operands and preprograms context 1's srcA base
// with the H_16 tile's L1 address. Call once during init, after
// llk_unpack_hw_configure. H_16 must stay resident at this address for
// the program's lifetime. Operand order: operandA = h16, operandB = input.
//
// Both unpackers run on h16's single-face config, and the input unpack
// reads 128 contiguous datums from the read pointer as 8 rows of 16. That
// is the logical 1x128 vector only if the input tile is a single face or
// its faces are one row tall (e.g. 4x[1,32]); taller multi-face tiles are
// stored face-major, which would silently permute the vector.
inline void llk_unpack_hadamard_h128_init(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t h16_tile_index) {
    SAN_HOOK(unsupported());
    const std::uint32_t operandA_id = get_operand_id(operandA);  // h16
    const std::uint32_t operandB_id = get_operand_id(operandB);  // input
    LLK_ASSERT(get_operand_num_faces(operandA_id) == 1, "Hadamard H128 unpack requires single-face h16 operand");
    LLK_ASSERT(
        get_operand_num_faces(operandB_id) == 1 || get_operand_face_r_dim(operandB_id) == 1,
        "Hadamard H128 unpack requires a single-face or one-row-face input operand");
    // The math path (fidelity phases, FP16B normalize) is bf16-only.
    LLK_ASSERT(
        get_operand_src_format(operandA_id) == (std::uint32_t)DataFormat::Float16_b,
        "Hadamard H128 requires a bfloat16 h16 operand");
    LLK_ASSERT(
        get_operand_src_format(operandB_id) == get_operand_src_format(operandA_id),
        "Hadamard H128 unpack requires the input format to match h16");

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
    SAN_HOOK(unsupported());
    const std::uint32_t operandA_id = get_operand_id(operandA);  // h16
    const std::uint32_t operandB_id = get_operand_id(operandB);  // input

    const std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    const std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;

    const std::uint32_t tile_size_a = get_local_cb_interface(operandA_id).fifo_page_size;
    const std::uint32_t tile_size_b = get_local_cb_interface(operandB_id).fifo_page_size;

    // The input unpack reads 128 datums from the tile address.
    LLK_ASSERT(
        get_local_cb_interface(operandB_id).fifo_rd_ptr + tile_size_b * tile_index_b +
                (SCALE_DATUM_SIZE(get_operand_src_format(operandB_id), 128) >> cb_addr_shift) <=
            get_local_cb_interface(operandB_id).fifo_limit,
        "Hadamard H128 input read exceeds CB boundary");

    WAYPOINT("UPHW");
    _llk_unpack_hadamard_h128_(base_address_a, base_address_b, tile_index_a, tile_index_b, tile_size_a, tile_size_b);
    WAYPOINT("UPHD");
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "experimental/llk_unpack_dest_srca_mm.h"
#include "llk_unpack_common_api.h"

// Unpack side of DEST_SRCA_MM (experimental/llk_unpack_dest_srca_mm.h): SrcA handed to the math thread, in0 to SrcB.

inline void llk_unpack_dest_srca_mm_init(const std::uint32_t in0_operand) {
    _llk_unpack_dest_srca_mm_init_(get_operand_num_faces(get_operand_id(in0_operand)));
}

inline void llk_unpack_dest_srca_mm(const std::uint32_t in0_operand, const std::uint32_t in0_tile_index) {
    const std::uint32_t operand_id = get_operand_id(in0_operand);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    _llk_unpack_dest_srca_mm_(base_address + get_local_cb_interface(operand_id).fifo_page_size * in0_tile_index);
}

// The page at tile_index of page_operand: its first `codes` 4-bit block-float datums (exponent exp) and `words` 32-bit
// words from byte words_offset.
inline void llk_unpack_dest_srca_mm_codes_page(
    const std::uint32_t page_operand,
    const std::uint32_t tile_index,
    const std::uint32_t codes,
    const std::uint32_t exp,
    const std::uint32_t words_offset,
    const std::uint32_t words) {
    const std::uint32_t operand_id = get_operand_id(page_operand);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    const std::uint32_t x_end = get_operand_face_r_dim(operand_id) * FACE_C_DIM - 1;
    _llk_unpack_dest_srca_mm_codes_page_(
        base_address + get_local_cb_interface(operand_id).fifo_page_size * tile_index,
        codes,
        exp,
        words_offset,
        words,
        x_end);
}

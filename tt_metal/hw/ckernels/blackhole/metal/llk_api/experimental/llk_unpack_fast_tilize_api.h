// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_common_api.h"
#include "experimental/llk_unpack_fast_tilize.h"

/*************************************************************************
 * LLK UNPACK FAST TILIZE SRC A (BH)
 *************************************************************************/

inline void llk_unpack_fast_tilize_init(
    const std::uint32_t operand, std::uint32_t full_dim, std::uint32_t init_unit_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_unpack_fast_tilize_init_(unpack_dst_format[operand_id], full_dim, init_unit_dim);
}

template <bool is_fp32_dest_acc_en>
inline void llk_unpack_fast_tilize_uninit() {
    _llk_unpack_fast_tilize_uninit_<is_fp32_dest_acc_en>();
}

inline void llk_unpack_fast_tilize_reinit_xdim(const std::uint32_t unit_dim) {
    _llk_unpack_fast_tilize_reinit_xdim_(unit_dim);
}

inline void llk_unpack_fast_tilize_block(
    const std::uint32_t operand,
    const std::uint32_t tile_index,
    const std::uint32_t unit_dim,
    const std::uint32_t col_start = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t src_format = unpack_src_format[operand_id];

    // Fold tile_index + col_start into the base address so the block function
    // always runs with Y=0 and no INCADCXY positioning loop.
    // Offset in source datums: (tile_index + col_start) * TILE_C_DIM columns.
    // SCALE_DATUM_SIZE converts datum count to bytes; >>4 to 16-byte L1 units.
    const std::uint32_t col_datum_offset = (tile_index + col_start) * TILE_C_DIM;
    const std::uint32_t base_address =
        (get_local_cb_interface(operand_id).fifo_rd_ptr - 1) + (SCALE_DATUM_SIZE(src_format, col_datum_offset) >> 4);

    _llk_unpack_fast_tilize_block_(base_address, tile_index, src_format, unit_dim, num_faces);
}

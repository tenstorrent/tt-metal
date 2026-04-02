// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_common_api.h"
#include "llk_unpack_unary_operand.h"
#include "experimental/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK — single-operand unary unpack (UNP_A / UNP_B / UNP_DEST)
 * Same operand / tile_index pattern as llk_unpack_AB_api.h.
 *************************************************************************/

/**
 * @brief Program unpack MOP for one unary operand (any unpack resource).
 * @param operand Logical dataflow buffer / CB id (resolved with get_operand_id).
 * @see tt_llk_quasar llk_unpack_unary_operand.h
 */
template <
    std::uint32_t UNP_SEL,
    bool TRANSPOSE_EN,
    bool IS_32b_DEST_EN,
    EltwiseBinaryReuseDestType reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_unpack_unary_operand_init(
    const std::uint32_t operand, const std::uint32_t num_tiles = NUM_TILES, const std::uint32_t num_faces = NUM_FACES) {
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_unpack_unary_operand_init_<UNP_SEL, TRANSPOSE_EN, IS_32b_DEST_EN, reuse_dest>(
        operand_id, num_tiles, num_faces);
}

/**
 * @brief Run unpack for one tile; L1 index = DFB read cursor + tile_index (same as llk_unpack_AB).
 */
template <std::uint32_t UNP_SEL, EltwiseBinaryReuseDestType reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_unpack_unary_operand_tile(const std::uint32_t operand, const std::uint32_t tile_index) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t l1_tile_idx = g_dfb_interface[operand_id].rd_entry_idx + tile_index;
    _llk_unpack_unary_operand_<UNP_SEL, reuse_dest>(l1_tile_idx);
}

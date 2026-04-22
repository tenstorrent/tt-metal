// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_common_api.h"
#include "llk_unpack_unary_operand.h"

/*************************************************************************
 * LLK UNPACK — single-operand unary unpack (UNP_A / UNP_B / UNP_DEST)
 *************************************************************************/

/**
 * @brief Program unpack MOP for one unary operand (any unpack resource).
 * @param operand Logical dataflow buffer / CB id (resolved with get_operand_id).
 * @param num_tiles Outer MOP tile count (default NUM_TILES). Face count is get_operand_num_faces(operand_id), like
 *                  Wormhole/Blackhole `llk_unpack_A_init` (num_faces is not a public argument there either). Override
 *                  via `_llk_unpack_unary_operand_init_` in tt_llk_quasar if needed (e.g. LLK tests).
 * @see tt_llk_quasar llk_unpack_unary_operand.h
 */
template <
    std::uint32_t UNP_SEL,
    bool TRANSPOSE_EN,
    bool IS_32b_DEST_EN,
    EltwiseBinaryReuseDestType reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_unpack_unary_operand_init(const std::uint32_t operand, const std::uint32_t num_tiles = NUM_TILES) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    _llk_unpack_unary_operand_init_<UNP_SEL, TRANSPOSE_EN, IS_32b_DEST_EN, reuse_dest>(
        operand_id, num_tiles, num_faces);
}

/**
 * @brief Run unpack for one tile; L1 index = DFB read cursor + tile_index (same as llk_unpack_AB).
 */
template <std::uint32_t UNP_SEL, EltwiseBinaryReuseDestType reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_unpack_unary_operand_tile(const std::uint32_t operand, const std::uint32_t tile_index) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const auto& local_dfb = g_dfb_interface[operand_id];
    const std::uint32_t l1_tile_idx = local_dfb.tc_slots[local_dfb.tc_idx].rd_entry_idx + tile_index;
    _llk_unpack_unary_operand_<UNP_SEL, reuse_dest>(l1_tile_idx);
}

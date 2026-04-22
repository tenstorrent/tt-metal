// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_common_api.h"
#include "llk_unpack_unary_broadcast_operands.h"

/*************************************************************************
 * LLK UNPACK — unary eltwise with scalar / row / column broadcast (Quasar)
 *************************************************************************/

/**
 * @brief Initialize unpack MOP for unary broadcast (scalar / row / column).
 * @param operand Logical dataflow buffer id for the unpack source.
 * @param num_tiles Outer MOP loop count (typically 1 per Metal unary_bcast init).
 */
template <std::uint32_t UNP_SEL, BroadcastType BROADCAST_TYPE, bool unpack_to_dest, bool is_fp32_dest_acc_en>
inline void llk_unpack_unary_broadcast_operands_init(const std::uint32_t operand, const std::uint32_t num_tiles) {
    static_assert(
        BROADCAST_TYPE != BroadcastType::NONE, "Unary broadcast unpack requires a broadcast dimension (not NONE)");

    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_unpack_unary_broadcast_operands_init_<UNP_SEL, BROADCAST_TYPE, unpack_to_dest, is_fp32_dest_acc_en>(
        operand_id, num_tiles);
}

/**
 * @brief Run unpack bank0 MOP for one tile (L1 = rd_entry_idx + tile_index).
 */
template <std::uint32_t UNP_SEL, bool unpack_to_dest>
inline void llk_unpack_unary_broadcast_operands(const std::uint32_t operand, const std::uint32_t tile_index) {
    static_assert(
        unpack_to_dest || (UNP_SEL == p_unpacr::UNP_B), "UNP_SEL must be p_unpacr::UNP_B when unpack_to_dest is false");

    const std::uint32_t operand_id = get_operand_id(operand);
    const auto& local_dfb = g_dfb_interface[operand_id];
    const std::uint32_t l1_tile_index = local_dfb.tc_slots[local_dfb.tc_idx].rd_entry_idx + tile_index;
    _llk_unpack_unary_broadcast_operands_<UNP_SEL, unpack_to_dest>(l1_tile_index);
}

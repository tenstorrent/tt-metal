// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_unary_operand.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK A
 *************************************************************************/

template <uint32_t UNP_SEL, bool TRANSPOSE_EN, bool IS_32b_DEST_EN>
inline void llk_unpack_A_init(const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);

    _llk_unpack_unary_operand_init_<UNP_SEL, TRANSPOSE_EN, IS_32b_DEST_EN>(operand_id, 1 /*num_tiles_per_unpack*/);
}

template <uint32_t UNP_SEL>
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    std::uint32_t operand_id = get_operand_id(operand);
    // Use fifo_rd_tile_idx which tracks how many tiles from CB base the read pointer is
    std::uint32_t l1_tile_index = get_local_cb_interface(operand_id).fifo_rd_tile_idx + tile_index;

    WAYPOINT("UPAW");
    _llk_unpack_unary_operand_<UNP_SEL>(l1_tile_index);
    WAYPOINT("UPAD");
}

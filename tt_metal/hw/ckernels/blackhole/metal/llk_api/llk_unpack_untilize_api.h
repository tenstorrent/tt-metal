// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_untilize.h"

/*************************************************************************
 * LLK UNPACK UNTILIZE
 *************************************************************************/

inline void llk_unpack_untilize_mop_config() { _llk_unpack_untilize_mop_config_(); }

inline void llk_unpack_untilize_init(std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = 1;

    _llk_unpack_untilize_init_(
        unpack_dst_format[operand_id], get_local_cb_interface(operand_id).fifo_page_size, face_r_dim);
}

inline void llk_unpack_untilize_uninit(const std::uint32_t operand, const std::uint32_t face_r_dim = FACE_R_DIM) {
    std::uint32_t operand_id = get_operand_id(operand);
    WAYPOINT("UPUW");
    _llk_unpack_untilize_uninit_((uint)unpack_dst_format[operand_id], face_r_dim);
    WAYPOINT("UPUD");
}

template <bool first_pass = true>
inline void llk_unpack_untilize_pass(std::uint32_t operand, std::uint32_t block_tile_cols) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;

    _llk_unpack_untilize_pass_<first_pass>(base_address, block_tile_cols);
}

inline void llk_unpack_untilize(std::uint32_t operand, std::uint32_t block_c_tiles) {
    WAYPOINT("UPUW");
    llk_unpack_untilize_pass<true>(operand, block_c_tiles);
    llk_unpack_untilize_pass<false>(operand, block_c_tiles);
    WAYPOINT("UPUD");
}

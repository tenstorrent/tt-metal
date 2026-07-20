// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_untilize.h"
#include "llk_unpack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK UNPACK UNTILIZE
 *
 * DEPRECATED: The unpack-based untilize path has poor performance and is
 * deprecated in favor of pack_untilize (see llk_pack_untilize_api.h). These
 * wrappers are retained only for the legacy `untilize_init/block/uninit`
 * compute API and are scheduled for removal; see tt-metal#22904.
 *************************************************************************/

inline void llk_unpack_untilize_init(std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = 1;

    _llk_unpack_untilize_init_(
        unpack_dst_format[operand_id], get_local_cb_interface(operand_id).fifo_page_size, face_r_dim);
}

inline void llk_unpack_untilize_uninit() {
    WAYPOINT("UPUW");
    _llk_unpack_untilize_uninit_();
    WAYPOINT("UPUD");
}

template <bool first_pass = true>
inline void llk_unpack_untilize_pass(std::uint32_t operand, std::uint32_t block_tile_cols) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;

    llk::san::unpack_operand_check(
        llk::san::IGNORE,
        unpack_src_format[operand_id],
        llk::san::IGNORE,
        unpack_dst_format[operand_id],
        llk::san::IGNORE,
        get_operand_face_r_dim(operand_id),
        llk::san::IGNORE,
        get_operand_num_faces(operand_id),
        llk::san::IGNORE);

    _llk_unpack_untilize_pass_<first_pass>(base_address, block_tile_cols);
}

inline void llk_unpack_untilize(std::uint32_t operand, std::uint32_t block_c_tiles) {
    WAYPOINT("UPUW");
    llk_unpack_untilize_pass<true>(operand, block_c_tiles);
    llk_unpack_untilize_pass<false>(operand, block_c_tiles);
    WAYPOINT("UPUD");
}

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

inline void llk_unpack_untilize_init(std::uint32_t operand) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = 1;

    _llk_unpack_untilize_init_(
        unpack_dst_format[operand_id], get_local_cb_interface(operand_id).fifo_page_size, face_r_dim);
}

inline void llk_unpack_untilize_uninit() {
    SAN_HOOK(unsupported());
    WAYPOINT("UPUW");
    _llk_unpack_untilize_uninit_();
    WAYPOINT("UPUD");
}

template <bool first_pass = true>
inline void llk_unpack_untilize_pass(std::uint32_t operand, std::uint32_t block_tile_cols) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;

    _llk_unpack_untilize_pass_<first_pass>(base_address, block_tile_cols);
}

inline void llk_unpack_untilize(std::uint32_t operand, std::uint32_t block_c_tiles) {
    SAN_HOOK(unsupported());
    WAYPOINT("UPUW");
    llk_unpack_untilize_pass<true>(operand, block_c_tiles);
    llk_unpack_untilize_pass<false>(operand, block_c_tiles);
    WAYPOINT("UPUD");
}

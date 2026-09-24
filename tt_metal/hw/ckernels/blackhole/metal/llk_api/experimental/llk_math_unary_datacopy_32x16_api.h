// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "sanitizer/api.h"

__attribute__((always_inline)) inline void llk_math_eltwise_unary_datacopy_to_dest_32x16(
    std::uint32_t dst_index, std::uint32_t operand) {
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x16>()), "");
    const std::uint32_t operand_id = get_operand_id(operand);
    const bool unpack_to_dest = is_32bit_input(unpack_src_format[operand_id], unpack_dst_format[operand_id]);
    LLK_ASSERT(unpack_to_dest, "32x16 unpack-to-dest requires a 32-bit operand format");
    if (!unpack_to_dest) {
        return;
    }
#ifdef SAN_HOOK
    SAN_HOOK(unsupported());
#endif
    math::math_unpack_to_dest_math_ready();
    math::set_dst_write_addr<DstTileShape::Tile32x16, UnpackDestination::DestReg>(dst_index);
    math::math_unpack_to_dest_tile_ready();

    // Blackhole can lose an unpack-to-dest zero-flag clear when packer ZEROACC issues in the same cycle.
    const std::uint32_t local_tile = dst_index & 7;
    TT_ZEROACC(p_zeroacc::CLR_16, 1, 1, ADDR_MOD_3, 2 * local_tile);
    TT_ZEROACC(p_zeroacc::CLR_16, 1, 1, ADDR_MOD_3, 2 * local_tile + 1);
}

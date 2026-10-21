// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "experimental/llk_math_eltwise_unary_datacopy_custom.h"
#include "sanitizer/api.h"

/**
 * @brief Math side of copy_tile_custom: copy one 32x32 tile from SrcA into DEST without a MOP.
 *
 * @param operand: The copied CB. Its unpack destination format (the format the copied datums have in SrcA) selects
 *                 the copy's Src zero-flag policy, as copy_init does.
 * @param dst_index: Tile index in DEST that the copy is written to.
 * @note Leaves the Src zero-substitution flag at the copy policy (keep; fp8 sources flushed). The next FPU op must
 *       re-run its init / reinit, which re-applies the operand-driven default.
 */
inline void llk_math_eltwise_unary_datacopy_custom(const std::uint32_t operand, const std::uint32_t dst_index) {
    SAN_HOOK(unsupported());
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()), "");
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_eltwise_unary_datacopy_custom_(dst_index, get_operand_dst_format(operand_id));
}

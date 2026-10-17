// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "experimental/llk_math_eltwise_unary_datacopy_custom.h"
#include "sanitizer/api.h"

// operand is the copied CB: its unpack destination format selects the copy's Src zero-flag policy, as copy_init does.
inline void llk_math_eltwise_unary_datacopy_custom(const std::uint32_t operand, const std::uint32_t dst_index) {
    SAN_HOOK(unsupported());
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()), "");
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_eltwise_unary_datacopy_custom_(dst_index, get_operand_dst_format(operand_id));
}

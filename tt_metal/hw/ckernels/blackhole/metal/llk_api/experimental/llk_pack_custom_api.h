// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "experimental/llk_pack_custom.h"

/*************************************************************************
 * LLK PACK NO MOP
 *************************************************************************/

// WARNING: Experimental API for SDPA optimizations only.
// This header has no corresponding tests in the llk-test infrastructure.
// Do not use outside of SDPA optimization workflows.

template <bool is_fp32_dest_acc_en, bool out_of_order_output = false>
inline void llk_pack_no_mop(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);

    std::uint32_t pack_tile_addr = get_output_tile_address<out_of_order_output, false>(output_id, output_tile_index);

    LLK_ASSERT((tile_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");
    _llk_pack_no_mop_<DST_SYNC_MODE, is_fp32_dest_acc_en>(tile_index, pack_tile_addr);
}

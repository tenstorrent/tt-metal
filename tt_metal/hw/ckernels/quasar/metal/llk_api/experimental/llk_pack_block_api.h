// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_pack_tile_api.h"

// The MOP names an output descriptor. Reinitialize when the DFB or tile shape changes,
// but not when only num_tiles changes. Normal pack hardware configuration must precede this.
inline void llk_pack_block_contiguous_mop_config(const std::uint32_t output) { llk_pack_init(output); }

template <bool is_fp32_dest_acc_en>
inline void llk_pack_block_contiguous(
    const std::uint32_t tile_index, const std::uint32_t output, const std::uint32_t num_tiles) {
    static_assert(is_fp32_dest_acc_en == DST_ACCUM_MODE, "Pack DEST mode must match the configured accumulation mode");
    LLK_ASSERT(num_tiles >= 1, "Contiguous pack requires at least one tile");
    LLK_ASSERT(
        get_local_dfb_interface(get_output_id(output)).stride_size_tiles == 1,
        "Contiguous pack requires consecutive output entries");
    llk_pack_block(tile_index, output, num_tiles);
}

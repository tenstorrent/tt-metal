// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tilize variant that exercises tilize_block with a NONZERO input_tile_index.
//
// Unlike tilize.cpp (which tilizes one tile-row at a time with input_tile_index=0 and advances the
// input via pop_front), this kernel loads the whole block once and tilizes each tile-row using
// input_tile_index = b * per_core_block_tile_cnt, without popping between rows. That drives the
// cross-tile-row stride term in llk_unpack_tilize_block, so the test can distinguish the fixed
// operand-derived stride from the old hardcoded TILE_R_DIM.

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    tilize_init(dfb::in, per_core_block_tile_cnt, dfb::out);

    const uint32_t total_tiles = per_core_block_cnt * per_core_block_tile_cnt;
    dfb_in.wait_front(total_tiles);
    dfb_out.reserve_back(total_tiles);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        const uint32_t tile_index = b * per_core_block_tile_cnt;
        tilize_block(dfb::in, per_core_block_tile_cnt, dfb::out, tile_index, tile_index);
    }

    dfb_in.pop_front(total_tiles);
    dfb_out.push_back(total_tiles);

    tilize_uninit(dfb::in, dfb::out);
}

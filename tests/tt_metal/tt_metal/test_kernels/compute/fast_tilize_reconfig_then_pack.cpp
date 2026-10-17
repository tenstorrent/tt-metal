// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Blackhole fast tilize with a same-format pack_reconfig_data_format between fast_tilize_init and the blocks,
// then fast_tilize_uninit and a normal pack. The fast-tilize init programs its own packer strides (Y = 64 face
// rows, Z = one face row, W = two face rows); the reconfig must leave them, a reconfig that resets them to the
// Default layout scrambles every tile. All block rows but the last are fast-tilized; after the uninit the last
// block row is copied through DEST tile by tile and packed with the Default pack, an identity on its bytes
// (the row-major data is read and written as tiles), which checks the strides the uninit restored. Width >= 2
// (width 1 falls back to the regular tilize), rows >= 2.
void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);
    static_assert(per_core_block_cnt >= 2, "needs a block row for the normal pack after the uninit");
    static_assert(per_core_block_tile_cnt >= 2, "width 1 falls back to the regular tilize");

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);

    fast_tilize_init(dfb::in, per_core_block_tile_cnt, dfb::out);
    pack_reconfig_data_format(dfb::out);

    for (uint32_t b = 0; b + 1 < per_core_block_cnt; ++b) {
        dfb_in.wait_front(per_core_block_tile_cnt);
        dfb_out.reserve_back(per_core_block_tile_cnt);
        fast_tilize_block(dfb::in, per_core_block_tile_cnt, dfb::out);
        dfb_in.pop_front(per_core_block_tile_cnt);
        dfb_out.push_back(per_core_block_tile_cnt);
    }

    fast_tilize_uninit(dfb::in, dfb::out, per_core_block_tile_cnt);

    // Normal pack after the uninit: identity copy of the last block row.
    copy_init(dfb::in);
    dfb_in.wait_front(per_core_block_tile_cnt);
    dfb_out.reserve_back(per_core_block_tile_cnt);
    tile_regs_acquire();
    for (uint32_t tile = 0; tile < per_core_block_tile_cnt; ++tile) {
        copy_tile(dfb::in, tile, tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t tile = 0; tile < per_core_block_tile_cnt; ++tile) {
        pack_tile(tile, dfb::out);
    }
    tile_regs_release();
    dfb_in.pop_front(per_core_block_tile_cnt);
    dfb_out.push_back(per_core_block_tile_cnt);
}

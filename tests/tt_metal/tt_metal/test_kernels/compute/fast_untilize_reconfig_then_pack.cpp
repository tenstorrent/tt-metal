// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/experimental/fast_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Blackhole fast untilize with a same-format pack_reconfig_data_format between fast_untilize_init and the blocks,
// then fast_untilize_uninit and a normal pack. The fast-untilize init programs its own packer strides (channel-0
// Y/Z/W for its block/phase scheme and, for a row wider than one 2/3/4-tile chunk, the channel-1 Y stride that
// advances the L1 row); the reconfig must leave them, a reconfig that resets them to the Default layout (and
// zeroes the channel-1 stride) scrambles the output. All block rows but the last are fast-untilized; after the
// uninit the last block row is copied through DEST tile by tile and packed with the Default pack (identity on
// its tiles), which checks the strides the uninit restored. Rows >= 2; a width above 4 takes the strided path.
void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t full_ct_dim = get_arg(args::per_core_block_tile_cnt);
    static_assert(per_core_block_cnt >= 2, "needs a block row for the normal pack after the uninit");

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);

    fast_untilize_init<full_ct_dim>(dfb::in, dfb::out);
    pack_reconfig_data_format(dfb::out);

    for (uint32_t r = 0; r + 1 < per_core_block_cnt; ++r) {
        dfb_in.wait_front(full_ct_dim);
        dfb_out.reserve_back(full_ct_dim);
        fast_untilize_block<full_ct_dim>(dfb::in, dfb::out);
        dfb_in.pop_front(full_ct_dim);
        dfb_out.push_back(full_ct_dim);
    }

    fast_untilize_uninit<full_ct_dim>(dfb::out);

    // Normal pack after the uninit: identity copy of the last block row.
    copy_init(dfb::in);
    dfb_in.wait_front(full_ct_dim);
    dfb_out.reserve_back(full_ct_dim);
    tile_regs_acquire();
    for (uint32_t tile = 0; tile < full_ct_dim; ++tile) {
        copy_tile(dfb::in, tile, tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t tile = 0; tile < full_ct_dim; ++tile) {
        pack_tile(tile, dfb::out);
    }
    tile_regs_release();
    dfb_in.pop_front(full_ct_dim);
    dfb_out.push_back(full_ct_dim);
}

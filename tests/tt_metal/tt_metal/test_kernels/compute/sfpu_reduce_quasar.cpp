// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t block_ct_dim = get_arg(args::block_ct_dim);
    constexpr uint32_t block_rt_dim = get_arg(args::block_rt_dim);
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);

    constexpr uint32_t tiles_per_block = block_ct_dim * block_rt_dim;

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    unary_op_init_common(dfb_in.get_id(), dfb_out.get_id());

    for (uint32_t block = 0; block < num_blocks; ++block) {
        sfpu_reduce_init<REDUCE_POOL_TYPE, REDUCE_FORMAT>();

        tile_regs_acquire();
        dfb_in.wait_front(tiles_per_block);

        // Gather the block into Dest, one tile per Dest tile. This only places the tiles correctly
        // when the operand unpacks to SrcA: unpacking straight to Dest skips the math-side datacopy
        // loop that advances the Dest index, and every tile lands on Dest tile 0 instead. The host
        // picks the mode, so Int32 -- which has to unpack to Dest -- stays single-tile.
        copy_block(dfb_in.get_id(), 0, 0, tiles_per_block);

#ifdef REDUCE_AXIS_ROW
        sfpu_reduce<REDUCE_POOL_TYPE, REDUCE_FORMAT, ReduceDim::REDUCE_ROW>(0, block_ct_dim, block_rt_dim);
#else
        for (uint32_t tile = 0; tile < tiles_per_block; ++tile) {
            sfpu_reduce<REDUCE_POOL_TYPE, REDUCE_FORMAT, ReduceDim::REDUCE_COL>(tile);
        }
#endif

        tile_regs_commit();
        tile_regs_wait();

        // Pack every tile of the block, even though on the row axis only the first one carries a
        // result: packing fewer tiles than the acquired Dest section holds hangs the device.
        for (uint32_t tile = 0; tile < tiles_per_block; ++tile) {
            dfb_out.reserve_back(1);
            pack_tile(tile, dfb_out.get_id());
            dfb_out.push_back(1);
        }

        dfb_in.pop_front(tiles_per_block);
        tile_regs_release();
    }
}

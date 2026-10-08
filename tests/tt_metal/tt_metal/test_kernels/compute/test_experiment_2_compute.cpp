// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/kernel_thread_globals.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t tiles_per_neo = get_arg(args::tiles_per_neo);  // rt arg per neo cluster

    const uint32_t num_neo_tensix = get_num_threads();
    const uint32_t my_neo_id = get_my_thread_id();
    const uint32_t tile_count = tiles_per_neo / num_neo_tensix + (my_neo_id < (tiles_per_neo % num_neo_tensix) ? 1 : 0);

    // in0 * in1 + in2 = out
    DataflowBuffer dfb_in0(dfb::in_0);
    DataflowBuffer dfb_in1(dfb::in_1);
    DataflowBuffer dfb_in2(dfb::in_2);
    DataflowBuffer dfb_out(dfb::out);

    uint32_t dst_tiles = 1;
    uint32_t dst_tile_idx = 0;

    compute_kernel_hw_startup(dfb_in0.get_id(), dfb_in1.get_id(), dfb_out.get_id());
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        dfb_in0.wait_front(dst_tiles);
        dfb_in1.wait_front(dst_tiles);
        dfb_in2.wait_front(dst_tiles);
        tile_regs_acquire();

        copy_init(dfb_in2.get_id());
        copy_tile(dfb_in2.get_id(), dst_tile_idx, dst_tile_idx);

        relu_tile_init();
        relu_tile(dst_tile_idx);

        mul_init(dfb_in0.get_id(), dfb_in1.get_id(), /* acc_to_dest= */ true);
        mul_tiles(dfb_in0.get_id(), dfb_in1.get_id(), dst_tile_idx, dst_tile_idx, dst_tile_idx);
        tile_regs_commit();

        dfb_in0.pop_front(dst_tiles);
        dfb_in1.pop_front(dst_tiles);
        dfb_in2.pop_front(dst_tiles);
        dfb_out.reserve_back(dst_tiles);

        tile_regs_wait();
        pack_tile(dst_tile_idx, dfb_out.get_id());

        tile_regs_release();
        dfb_out.push_back(dst_tiles);
    }

    dfb_in0.finish();
    dfb_in1.finish();
    dfb_in2.finish();
    dfb_out.finish();
}

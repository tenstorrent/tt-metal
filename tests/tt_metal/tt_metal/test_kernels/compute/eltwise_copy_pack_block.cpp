// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Exercises the uniform op_block Compute API surface (copy_block + pack_block), i.e. the renamed
// performant block paths that superseded copy_block_matmul_partials / pack_tile_block. Mirrors
// eltwise_copy_block_matmul_partials.cpp but issues a single block call per ublock instead of a
// loop of single-tile calls, so the golden (identity copy) is identical.
void kernel_main() {
    constexpr uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t num_single_transfer = get_arg(args::num_single_transfer);

    constexpr uint32_t outer_loop = num_tiles / num_single_transfer;

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);
    unary_op_init_common(dfb::in, dfb::out);

    for (uint32_t b = 0; b < outer_loop; ++b) {
        dfb_in.wait_front(num_single_transfer);
        dfb_out.reserve_back(num_single_transfer);
        tile_regs_acquire();
        tile_regs_wait();

        copy_block(dfb::in, /* start_in_tile_index */ 0, /* start_dst_tile_index */ 0, num_single_transfer);

        pack_block(/* ifrom_dst */ 0, dfb::out, num_single_transfer);

        tile_regs_commit();
        tile_regs_release();
        dfb_in.pop_front(num_single_transfer);
        dfb_out.push_back(num_single_transfer);
    }
}

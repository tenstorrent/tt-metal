// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

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

        for (uint32_t i = 0; i < num_single_transfer; ++i) {
            copy_block_matmul_partials(dfb::in, i, i, 1);
        }

        pack_tile_block(0, dfb::out, num_single_transfer);

        tile_regs_commit();
        tile_regs_release();
        dfb_in.pop_front(num_single_transfer);
        dfb_out.push_back(num_single_transfer);
    }
}

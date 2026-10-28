// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multiplies each tile of in_a by the matching tile of in_b into out.

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_a(dfb::in_a);
    DataflowBuffer dfb_b(dfb::in_b);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in_a, dfb::in_b, dfb::out);
    mul_init(dfb::in_a, dfb::in_b);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_a.wait_front(1);
        dfb_b.wait_front(1);

        tile_regs_acquire();
        mul_tiles(dfb::in_a, dfb::in_b, 0, 0, 0);
        tile_regs_commit();

        dfb_a.pop_front(1);
        dfb_b.pop_front(1);

        dfb_out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dfb::out, 0);
        tile_regs_release();
        dfb_out.push_back(1);
    }
}

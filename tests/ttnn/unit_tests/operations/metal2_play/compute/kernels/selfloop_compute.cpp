// SPDX-License-Identifier: Apache-2.0
// PROBE G: SELF-LOOP DFB. dfb::resident_out is bound by THIS kernel as both PRODUCER and
// CONSUMER under one accessor name. It is `borrowed_from` the op's L1-resident output tensor, so
// the packer writes straight into the output buffer and there is no writer kernel to drain it --
// the compute kernel recycles its own credits.
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer in(dfb::in_tiles);
    DataflowBuffer resident(dfb::resident_out);  // one object, both roles

    compute_kernel_hw_startup(dfb::in_tiles, dfb::resident_out);
    copy_tile_init(dfb::in_tiles);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        in.wait_front(1);
        resident.reserve_back(1);

        tile_regs_acquire();
        copy_tile(dfb::in_tiles, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::resident_out, 0);
        tile_regs_release();

        in.pop_front(1);
        resident.push_back(1);
    }

    // Drain our own pushes so the DFB ends the program empty.
    resident.wait_front(num_tiles);
    resident.pop_front(num_tiles);
}

// SPDX-License-Identifier: Apache-2.0
// Plain per-tile copy compute kernel. Used by the ALIAS probe (in/out DFBs may share L1) and by
// the fp32 / unpack_modes probe (a bare copy is already a precision test: UnpackToSrc truncates
// the fp32 mantissa on the way through SrcA, UnpackToDest does not).
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer in(dfb::in_tiles);
    DataflowBuffer out(dfb::out_tiles);

    compute_kernel_hw_startup(dfb::in_tiles, dfb::out_tiles);
    copy_tile_init(dfb::in_tiles);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        in.wait_front(1);

        tile_regs_acquire();
        copy_tile(dfb::in_tiles, 0, 0);
        tile_regs_commit();

        // The unpack read of `in` has completed (tile_regs_commit/wait orders MATH before PACK), so
        // packing into an L1 region ALIASED with `in` is safe here. Nothing in the framework checks
        // this: alias_with shares the address space and gives no ordering whatsoever.
        in.pop_front(1);

        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dfb::out_tiles, 0);
        tile_regs_release();
        out.push_back(1);
    }
}

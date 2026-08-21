// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: compute-side DFB consumer that copies each tile straight through.
//
// The point of this kernel is not the arithmetic, which is an identity, but which hardware performs
// the wait. A data-movement consumer of a DataflowBuffer spins on the buffer's occupancy from a RISC.
// A compute consumer waits through the unpacker instead. Putting a compute kernel in the consumer
// position therefore exercises a different implementation of the same contract: wait_front(1) must not
// return until the producer's matching push_back(1).
//
// If the wait releases early, the unpacker reads a slot the producer has not filled, the packer writes
// that stale tile to the output buffer, and the mismatch reaches DRAM where the host can see it. So a
// plain copy is enough to detect the failure; anything more would only obscure where it came from.
//
// Runtime args:
//   arg 0: number of tiles to copy

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    unary_op_init_common(dfb::in0, dfb::out);
    copy_tile_init(dfb::in0);

    DataflowBuffer cb_in0(dfb::in0);
    DataflowBuffer cb_out(dfb::out);

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();

        cb_in0.wait_front(1);
        cb_out.reserve_back(1);
        copy_tile(dfb::in0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out);

        cb_in0.pop_front(1);
        cb_out.push_back(1);

        tile_regs_release();
    }
}

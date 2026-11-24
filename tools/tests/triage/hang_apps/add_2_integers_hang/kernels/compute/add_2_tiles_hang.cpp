// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/assert.h"

void kernel_main() {
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);

    // The following sequence of operations are compiled onto the 3 compute cores (Unpack, Math, Pack) in the Tensix
    // core. The work together to perform the addition of two input tiles and store the result in the output tile to the
    // output dataflow buffer. Which is then picked up by the writer kernel and written back to DRAM.

    // Metalium API Calls                              Involved Cores
    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);  // Unpack, Math, Pack
    add_init(dfb::in0, dfb::in1);                             // Unpack, Math

    // wait for a tile to be ready in the input dataflow buffers
    dfb_in0.wait_front(1);    // Unpack
    dfb_in1.wait_front(1);    // Unpack
    dfb_out.reserve_back(1);  // Pack

    // acquire 8 tile registers to perform the addition
    tile_regs_acquire();  // Math

    // Take data from dfb::in0 offset 0th page and
    // dfb::in1 offset 0th page. Add them together
    // and store the result in dfb::out (as
    // configured) offset 0th page.
    add_tiles(dfb::in0, dfb::in1, 0, 0, 0);  // Unpack, Math

    // signal the packer
    tile_regs_commit();  // Math
#if LIGHTWEIGHT_ASSERT_ENABLED
    // Intentional hang on a lightweight ASSERT (tt-exalens#1164); test_triage.py asserts it is on line 43.
    volatile uint32_t magic = 0, n = 0xDEADBEEF;
    ASSERT(magic == n);
#else
    // Intentional hang for triage testing; test_triage.py asserts this `ebreak` is on line 46.
    asm volatile("ebreak");
#endif

    // packer waits here
    tile_regs_wait();  // Pack
    // Copy the result from tile registers to the
    // output dataflow buffer (also called packing)
    pack_tile(0, dfb::out);  // Pack
    // packer releases
    tile_regs_release();  // Pack

    dfb_in0.pop_front(1);  // Unpack
    dfb_in1.pop_front(1);  // Unpack

    dfb_out.push_back(1);  // Pack
}

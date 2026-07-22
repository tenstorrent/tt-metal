// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) Tensix-side consumer for the single-DFB matrix
// sweep (DM → DFB → TRISC case).
//
// This kernel adds Tensix-as-consumer coverage that the DM-side consumer
// (dfb_consumer_2_0.cpp) doesn't reach. The DM consumer drains the DFB and
// NoC-writes data to DRAM; this kernel drains DFB credits on the Tensix side
// and calls finish(), exercising the Tensix wait_front / copy_tile / pop_front /
// finish path without coupling the test to a NoC write back-half. Per HW spec a
// copy (unpack) instruction must sit between wait_front and pop_front, so each
// entry is copied into the math dest register and then discarded (there is no
// output DFB).
//
// Flow per test invocation:
//   1. DM producer kernel writes data into the DFB L1 ring (NoC read from DRAM).
//   2. This kernel does wait_front + copy_tile + pop_front for
//      num_entries_per_consumer iterations, then dfb.finish().
//   3. Host verifies the program ran (DM→Tensix L1 verification is omitted).
//
// Bindings (set by host KernelSpec):
//   dfb::in — CONSUMER (host binds the same DFB the DM producer pushes to).

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);

    DataflowBuffer dfb(dfb::in);

    // HW requires a copy (unpack) instruction between wait_front and pop_front.
    // This kernel has no output DFB, so configure unpack + pack hw against the
    // single input DFB (matches the production compute_kernel_hw_startup(in, out);
    copy_init(in)
    // shape; this is also what programs the buffer-descriptor table the UNPACR
    // reads) and discard the copied tile. copy_tile reads the entry into the
    // math dest register without mutating the DFB L1 ring. The acquire_dst /
    // release_dst pair alone balances the MATH<->PACK dest handshake (no
    // pack_tile required, and packing into the input ring would race the producer).
    compute_kernel_hw_startup(dfb.get_id(), dfb.get_id());
    copy_init(dfb.get_id());

    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; ++tile_id) {
        acquire_dst();
        dfb.wait_front(1);
        copy_tile(dfb.get_id(), 0, 0);
        dfb.pop_front(1);
        release_dst();
    }
    dfb.finish();
}

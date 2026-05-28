// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) Tensix-side consumer for the single-DFB matrix
// sweep (DM → DFB → TRISC case).
//
// This kernel adds Tensix-as-consumer coverage that the DM-side consumer
// (dfb_consumer_2_0.cpp) doesn't reach. The DM consumer drains the DFB and
// NoC-writes data to DRAM; this kernel just drains DFB credits and calls
// finish() — it never moves data anywhere. That exercises the Tensix-side
// wait_front / pop_front / finish path without coupling the test to a NoC
// write back-half.
//
// Flow per test invocation:
//   1. DM producer kernel writes data into the DFB L1 ring (NoC read from DRAM).
//   2. This kernel does wait_front + pop_front for num_entries_per_consumer
//      iterations, then dfb.finish().
//   3. Host reads L1 directly after the run and verifies the data is correct.
//
// Bindings (set by host KernelSpec):
//   dfb::in — CONSUMER (host binds the same DFB the DM producer pushes to).

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);

    DataflowBuffer dfb(dfb::in);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; ++tile_id) {
        dfb.wait_front(1);
        dfb.pop_front(1);
    }
    dfb.finish();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) Tensix-side producer for the single-DFB matrix
// sweep (TRISC → DFB → DM case).
//
// This kernel adds Tensix-as-producer coverage. Because TRISC compute kernels
// can't NoC-read from DRAM in this test setup, the host pre-fills the DFB's L1
// ring directly via WriteToDeviceL1 before the program launches. The kernel
// itself only does reserve_back / push_back — it posts the credits that say
// "one tile is available", which the downstream DM consumer waits on.
//
// Flow per test invocation:
//   1. Host pre-fills the DFB L1 ring with the input data.
//   2. This kernel calls reserve_back + push_back num_entries_per_producer
//      times, then dfb.finish().
//   3. The DM consumer drains those credits, reads from L1, and NoC-writes
//      out to DRAM.
//   4. Host reads DRAM to verify.
//
// Bindings (set by host KernelSpec):
//   dfb::out — PRODUCER (host pre-binds the same DFB the DM consumer reads).

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);

    DataflowBuffer dfb(dfb::out);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; ++tile_id) {
        dfb.reserve_back(1);
        dfb.push_back(1);
    }
    dfb.finish();
}

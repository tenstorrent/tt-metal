// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) Tensix-side BLOCKED producer (TRISC → DFB → DM case).
//
// Parallel to dfb_t6_producer_2_0.cpp, but posts credits at BLOCKED granularity:
// reserve_back(block_size) / push_back(block_size) once per block instead of per
// tile. As with the STRIDED Tensix producer, TRISC compute kernels can't NoC-read
// from DRAM in this test setup, so the host pre-fills the DFB's L1 ring before the
// program launches; this kernel only posts the credits the downstream DM consumer
// waits on. The block-ness here is purely the credit cadence (block_size at a time);
// the per-thread contiguous sub-ring layout is set up host-side (stride_in_entries=1).
//
// Bindings/CTAs (set by host KernelSpec):
//   dfb::out                  — PRODUCER
//   num_entries_per_producer  — total entries this thread produces
//   block_size                — tiles per block

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t block_size = get_arg(args::block_size);

    DataflowBuffer dfb(dfb::out);

    const uint32_t num_blocks = num_entries_per_producer / block_size;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        dfb.reserve_back(block_size);
        dfb.push_back(block_size);
    }
    dfb.finish();
}

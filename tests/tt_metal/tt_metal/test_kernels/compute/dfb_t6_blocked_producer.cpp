// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) Tensix-side BLOCKED producer (TRISC -> DFB -> DM).
// Parallel to dfb_t6_producer_2_0.cpp, but posts credits a block at a time. TRISC kernels
// can't NoC-read DRAM in this setup, so the host pre-fills the ring and this kernel only
// posts the credits the DM consumer waits on.

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

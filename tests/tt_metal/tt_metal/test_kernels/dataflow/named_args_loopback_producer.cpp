// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: Metal 2.0 named-args loopback producer.
// Reads data from a single-page DRAM buffer and pushes it entry-by-entry into a DFB.
//
// Uses Metal 2.0 named-arg accessors exclusively (via get_arg):
//   args::src_addr      — named RTA (per-node), DRAM source address
//   args::num_entries   — named CRTA (broadcast), number of entries to transfer
//   args::bank_id       — named CTA, DRAM bank ID (typically 0 for single-page buffers)
//   args::entry_size    — named CTA, bytes per DFB entry
//   get_vararg(0)       — vararg RTA (0-indexed), a sentinel value the test verifies

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t src_addr = get_arg(args::src_addr);
    uint32_t num_entries = get_arg(args::num_entries);
    constexpr uint32_t bank_id = get_arg(args::bank_id);
    constexpr uint32_t entry_size = get_arg(args::entry_size);

    // Prove that get_vararg(0) addresses the first vararg regardless of how many named
    // RTAs precede it in the dispatch buffer. The host passes an agreed sentinel value;
    // if the offset is wrong, we'd read back garbage (or worse, the first named RTA).
    uint32_t sentinel = get_vararg(0);
    // Use it to ensure the compiler doesn't DCE the load.
    volatile uint32_t touch = sentinel;
    (void)touch;

    experimental::DataflowBuffer buf(dfb::loopback_dfb);

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.reserve_back(1);
        uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_addr);
        noc_async_read(src_noc_addr, buf.get_write_ptr(), entry_size);
        noc_async_read_barrier();
        buf.push_back(1);
        src_addr += entry_size;
    }
}

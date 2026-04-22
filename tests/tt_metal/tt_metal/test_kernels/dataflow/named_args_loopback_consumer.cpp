// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: Metal 2.0 named-args loopback consumer.
// Reads data from a DFB and writes it to a single-page DRAM buffer.
//
// Uses Metal 2.0 named-arg accessors exclusively (via get_arg):
//   args::dst_addr      — named RTA (per-node), DRAM destination address
//   args::num_entries   — named CRTA (broadcast), number of entries to transfer
//   args::bank_id       — named CTA, DRAM bank ID
//   args::entry_size    — named CTA, bytes per DFB entry

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t dst_addr = get_arg(args::dst_addr);
    uint32_t num_entries = get_arg(args::num_entries);
    constexpr uint32_t bank_id = get_arg(args::bank_id);
    constexpr uint32_t entry_size = get_arg(args::entry_size);

    experimental::DataflowBuffer buf(dfb::loopback_dfb);

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.wait_front(1);
        uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dst_addr);
        noc_async_write(buf.get_read_ptr(), dst_noc_addr, entry_size);
        noc_async_write_barrier();
        buf.pop_front(1);
        dst_addr += entry_size;
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: DFB loopback consumer using DFB local accessor name.
// Reads data entry-by-entry from a DFB bound via the dfb::a_dfb_named_bob accessor name
// (from kernel_bindings_generated.h) and writes it to a single-page DRAM buffer.
//
// Runtime args:
//   arg 0: destination DRAM address
//   arg 1: DRAM bank ID
//   arg 2: number of entries to transfer

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_entries = get_arg_val<uint32_t>(2);

    // Construct the DataflowBuffer using the named accessor from kernel_bindings_generated.h
    experimental::DataflowBuffer buf(dfb::a_dfb_named_bob);
    uint32_t entry_size = buf.get_entry_size();

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.wait_front(1);
        uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dst_addr);
        noc_async_write(buf.get_read_ptr(), dst_noc_addr, entry_size);
        noc_async_write_barrier();
        buf.pop_front(1);
        dst_addr += entry_size;
    }
}

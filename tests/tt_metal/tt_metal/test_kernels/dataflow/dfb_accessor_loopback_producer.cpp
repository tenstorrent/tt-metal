// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: DFB loopback producer using DFB local accessor name.
// Reads data from a single-page DRAM buffer and pushes it entry-by-entry into a DFB
// bound via the dfb::my_local_dfb_name accessor name (from kernel_bindings_generated.h).
//
// Runtime args:
//   arg 0: source DRAM address
//   arg 1: DRAM bank ID
//   arg 2: number of entries to transfer

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_entries = get_arg_val<uint32_t>(2);

    // Construct the DataflowBuffer using the named accessor from kernel_bindings_generated.h
    experimental::DataflowBuffer buf(dfb::my_local_dfb_name);
    uint32_t entry_size = buf.get_entry_size();

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.reserve_back(1);
        uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_addr);
        noc_async_read(src_noc_addr, buf.get_write_ptr(), entry_size);
        noc_async_read_barrier();
        buf.push_back(1);
        src_addr += entry_size;
    }
}

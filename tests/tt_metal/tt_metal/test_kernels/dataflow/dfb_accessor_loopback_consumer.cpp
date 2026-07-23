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
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_entries = get_arg_val<uint32_t>(2);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_dst;
    DataflowBuffer buf(dfb::a_dfb_named_bob);
    uint32_t entry_size = buf.get_entry_size();

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.wait_front(1);
        noc.async_write(buf, dram_dst, entry_size, {}, {.bank_id = bank_id, .addr = dst_addr});
        noc.async_write_barrier();
        buf.pop_front(1);
        dst_addr += entry_size;
    }
}

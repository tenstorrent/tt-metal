// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Minimal DataflowBuffer consumer: wait for one entry, write it to DRAM, release it.
//
// This is the textbook drain loop and it is a data-movement kernel, which is what makes it the affected
// side of the contract. wait_front(1) must not return until the producer's matching push_back(1).
//
// Runtime args:
//   arg 0: destination DRAM address
//   arg 1: number of entries to copy

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_entries = get_arg_val<uint32_t>(1);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram;
    DataflowBuffer buf(dfb::ring);

    const uint32_t entry_size = buf.get_entry_size();

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.wait_front(1);
        noc.async_write(buf, dram, entry_size, {}, {.bank_id = 0, .addr = dst_addr});
        noc.async_write_barrier();
        buf.pop_front(1);
        dst_addr += entry_size;
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Minimal DataflowBuffer producer that fills each entry with two half-entry reads.
//
// The buffer is used exactly as documented: reserve one slot, fill it, barrier, announce it. The only
// thing unusual is that the fill takes two NoC reads instead of one, so the kernel issues twice as many
// transactions as the slots it announces. Both reads land inside the slot it reserved, and nothing else
// is touched.
//
// Replacing the two reads with one read of entry_size bytes is the control, and it passes everywhere.
//
// Runtime args:
//   arg 0: source DRAM address
//   arg 1: number of entries to copy

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_entries = get_arg_val<uint32_t>(1);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram;
    DataflowBuffer buf(dfb::ring);

    const uint32_t entry_size = buf.get_entry_size();
    const uint32_t half = entry_size / 2;

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.reserve_back(1);
        noc.async_read(dram, buf, half, {.bank_id = 0, .addr = src_addr}, {.offset_bytes = 0});
        noc.async_read(dram, buf, half, {.bank_id = 0, .addr = src_addr + half}, {.offset_bytes = half});
        noc.async_read_barrier();
        buf.push_back(1);
        src_addr += entry_size;
    }
}

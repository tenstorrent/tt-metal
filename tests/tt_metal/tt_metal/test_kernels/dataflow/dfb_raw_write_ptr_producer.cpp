// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: DFB producer that fills the buffer through a raw write pointer.
//
// Identical to dfb_accessor_loopback_producer.cpp except for one line: the NoC read targets
// `buf.get_write_ptr()` wrapped in a CoreLocalMem, rather than the DataflowBuffer itself. Both are
// documented ways to fill a DFB, and on Gen1 they are equivalent, but they differ in what the
// runtime sees: passing the buffer lets the transfer be attributed to the DFB, while passing an
// address does not.
//
// This is the shape real ported readers have, because they compute a per-token address and hand that
// to a helper. Paired with the DFB-object producer in the same test file, it isolates whether that
// difference alone changes the credit behaviour.
//
// Runtime args:
//   arg 0: source DRAM address
//   arg 1: DRAM bank ID
//   arg 2: number of entries to transfer

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_entries = get_arg_val<uint32_t>(2);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;
    DataflowBuffer buf(dfb::my_local_dfb_name);
    uint32_t entry_size = buf.get_entry_size();

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.reserve_back(1);
        uint32_t l1_write_addr = buf.get_write_ptr();
        noc.async_read(
            dram_src, CoreLocalMem<uint32_t>(l1_write_addr), entry_size, {.bank_id = bank_id, .addr = src_addr}, {});
        noc.async_read_barrier();
        buf.push_back(1);
        src_addr += entry_size;
    }
}

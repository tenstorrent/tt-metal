// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: the smallest producer that issues one extra NoC read before filling a DataflowBuffer.
//
// A direct transcription of the equivalent case in the indexed_fill investigation's repro, written
// with no scratchpad, no mode switch and no lambda, so that comparing it against
// dfb_scratchpad_scope_producer.cpp in mode 6 separates a difference in the kernel from a difference
// in how the host builds the program. The two are supposed to do the same thing.
//
// Runtime args:
//   arg 0: source DRAM address for the DFB entries
//   arg 1: DRAM bank ID
//   arg 2: number of DFB entries to push
//   arg 3: plain SRAM address for the extra read

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_entries = get_arg_val<uint32_t>(2);
    uint32_t plain_addr = get_arg_val<uint32_t>(3);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;
    DataflowBuffer buf(dfb::my_local_dfb_name);
    uint32_t entry_size = buf.get_entry_size();

    CoreLocalMem<volatile uint32_t> plain{static_cast<uintptr_t>(plain_addr)};
    noc.async_read(dram_src, plain, 32u, {.bank_id = bank_id, .addr = src_addr}, {.offset_bytes = 0});
    noc.async_read_barrier();

    for (uint32_t d = 0; d < 2000; d++) {
        asm volatile("" ::: "memory");
    }

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.reserve_back(1);
        noc.async_read(dram_src, buf, entry_size, {.bank_id = bank_id, .addr = src_addr}, {});
        noc.async_read_barrier();
        buf.push_back(1);
        src_addr += entry_size;
    }
}

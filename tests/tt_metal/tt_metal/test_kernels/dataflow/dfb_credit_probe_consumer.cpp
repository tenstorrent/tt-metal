// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: DFB consumer that records what each entry actually contained at the moment its grant
// was taken.
//
// Same DFB to DRAM loop as dfb_accessor_loopback_consumer.cpp, plus one sample per iteration: right
// after wait_front(1) returns, read the first word of the granted entry and store it. The host knows
// what the producer put in every entry, so it can check whether the entry the consumer was handed
// was actually filled yet.
//
// Why sample the entry rather than a counter the producer publishes: the entry arrives by NoC and is
// covered by the producer's async_read_barrier, so it is subject to exactly the ordering the DFB is
// supposed to guarantee. A counter written by a software store is not, because push_back posts its
// credit through an uncached overlay register while a plain store goes to cached SRAM, so a counter
// can read low for reasons that have nothing to do with the buffer. Sampling the payload removes
// that confound.
//
// Reading the entry through a volatile pointer is what the RISC can do here; the invalidate before it
// is required because the ring reuses slots, so a stale cached line for this address is possible.
//
// Runtime args:
//   arg 0: destination DRAM address
//   arg 1: DRAM bank ID
//   arg 2: number of entries to transfer
//   arg 3: SRAM address of the sample array (num_entries uint32s)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_entries = get_arg_val<uint32_t>(2);
    uint32_t samples_addr = get_arg_val<uint32_t>(3);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_dst;
    DataflowBuffer buf(dfb::consumer_dfb);
    uint32_t entry_size = buf.get_entry_size();

    volatile tt_l1_ptr uint32_t* samples = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(samples_addr);

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.wait_front(1);
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* entry = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf.get_read_ptr());
        samples[i] = entry[0];
        noc.async_write(buf, dram_dst, entry_size, {}, {.bank_id = bank_id, .addr = dst_addr});
        noc.async_write_barrier();
        buf.pop_front(1);
        dst_addr += entry_size;
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: plain DFB producer for the credit probe.
//
// Identical to dfb_accessor_loopback_producer.cpp; it exists only so the probe test can bind its own
// accessor name (producer_dfb) without perturbing the shared loopback kernels. Each entry it stages
// carries a distinct first word, which is what dfb_credit_probe_consumer.cpp samples.
//
// Runtime args:
//   arg 0: source DRAM address
//   arg 1: DRAM bank ID
//   arg 2: number of entries to transfer

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_entries = get_arg_val<uint32_t>(2);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;
    DataflowBuffer buf(dfb::producer_dfb);
    uint32_t entry_size = buf.get_entry_size();

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.reserve_back(1);
        noc.async_read(dram_src, buf, entry_size, {.bank_id = bank_id, .addr = src_addr}, {});
        noc.async_read_barrier();
        buf.push_back(1);
        src_addr += entry_size;
    }
}

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"
#include "tensix_types.h"

// DRAM to L1 read using stateful read API for optimal throughput.
void kernel_main() {
    uint32_t src_addr = get_arg(args::src_addr);
    uint32_t l1_addr = get_arg(args::l1_addr);

    constexpr uint32_t num_of_transactions = get_arg(args::num_transactions);
    constexpr uint32_t num_banks = get_arg(args::num_banks);
    constexpr uint32_t pages_per_bank = get_arg(args::pages_per_bank);
    constexpr uint32_t page_size_bytes = get_arg(args::page_size);
    constexpr uint32_t test_id = get_arg(args::test_id);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", num_banks * pages_per_bank * page_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    Noc noc(noc_index);
    UnicastEndpoint unicast_endpoint;
    AllocatorBank<AllocatorBankType::DRAM> dram_bank;

    uint32_t dst_addr = l1_addr;

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t n = 0; n < num_of_transactions; n++) {
            dst_addr = l1_addr;
            for (uint32_t bank_id = 0; bank_id < num_banks; bank_id++) {
                noc.set_async_read_state(dram_bank, page_size_bytes, {.bank_id = bank_id, .addr = src_addr});

                for (uint32_t i = 0; i < pages_per_bank; i++) {
                    noc.async_read_with_state(
                        dram_bank,
                        unicast_endpoint,
                        page_size_bytes,
                        {.bank_id = bank_id, .addr = src_addr + i * page_size_bytes},
                        {.addr = dst_addr});
                    dst_addr += page_size_bytes;
                }
            }
        }
        noc.async_read_barrier();
    }
}

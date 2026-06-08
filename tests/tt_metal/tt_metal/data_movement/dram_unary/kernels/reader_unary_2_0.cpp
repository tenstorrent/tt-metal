// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"

// DRAM to L1 read
void kernel_main() {
    // True compile-time constants
    constexpr uint32_t test_id = get_arg(args::test_id);
    constexpr uint32_t bytes_per_page = get_arg(args::bytes_per_page);
    constexpr uint32_t dram_addr = get_arg(args::dram_addr);
    constexpr uint32_t dram_channel = get_arg(args::dram_channel);
    constexpr uint32_t local_l1_addr = get_arg(args::l1_addr);

    uint32_t num_of_transactions = get_arg(args::num_of_transactions);
    uint32_t pages_per_transaction = get_arg(args::pages_per_transaction);
    uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    Noc noc(noc_index);
    UnicastEndpoint unicast_endpoint;
    Semaphore semaphore(sem::dram_sync);
    constexpr AllocatorBankType bank_type = AllocatorBankType::DRAM;

    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            noc.async_read(
                AllocatorBank<bank_type>(),
                unicast_endpoint,
                bytes_per_transaction,
                {
                    .bank_id = dram_channel,
                    .addr = dram_addr,
                },
                {
                    .addr = local_l1_addr,
                });
        }
        noc.async_read_barrier();
    }

    // Set the semaphore to indicate that the writer can proceed
    semaphore.set(1);

    DeviceTimestampedData("Test id", test_id);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
}

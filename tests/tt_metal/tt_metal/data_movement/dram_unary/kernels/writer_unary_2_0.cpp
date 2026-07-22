// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "internal/dataflow/dataflow_api_common.h"

// L1 to DRAM write
void kernel_main() {
    // True compile-time constants
    constexpr uint32_t test_id = get_arg(args::test_id);
    constexpr uint32_t bytes_per_page = get_arg(args::bytes_per_page);
    constexpr uint32_t dram_channel = get_arg(args::dram_channel);
    constexpr uint32_t local_l1_addr = get_arg(args::l1_addr);
    constexpr uint32_t virtual_channel = get_arg(args::vc);

    uint32_t num_of_transactions = get_arg(args::num_of_transactions);
    uint32_t pages_per_transaction = get_arg(args::pages_per_transaction);
    uint32_t dram_addr = get_arg(args::dram_addr);
    uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    Noc noc(noc_index);
    UnicastEndpoint unicast_endpoint;
    Semaphore semaphore(sem::dram_sync);
    constexpr AllocatorBankType bank_type = AllocatorBankType::DRAM;

    // Wait for semaphore to be set by the reader
    semaphore.down(1);

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            noc.async_write<NocOptions::CUSTOM_VC>(
                unicast_endpoint,
                AllocatorBank<bank_type>(),
                bytes_per_transaction,
                {
                    .addr = local_l1_addr,
                },
                {
                    .bank_id = dram_channel,
                    .addr = dram_addr,
                },
                NocOptVals{.vc = virtual_channel});
        }
        noc.async_write_barrier();
    }

    DeviceTimestampedData("Test id", test_id);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);

    DeviceTimestampedData("DRAM Channel", dram_channel);
    DeviceTimestampedData("Virtual Channel", virtual_channel);
}

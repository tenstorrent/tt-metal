// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/endpoints.h"
#include "experimental/noc_semaphore.h"
#include "internal/dataflow/dataflow_api_common.h"

// L1 to DRAM write
void kernel_main() {
    constexpr uint32_t test_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t pages_per_transaction = get_compile_time_arg_val(2);
    constexpr uint32_t bytes_per_page = get_compile_time_arg_val(3);
    constexpr uint32_t dram_addr = get_compile_time_arg_val(4);
    constexpr uint32_t dram_channel = get_compile_time_arg_val(5);
    constexpr uint32_t local_l1_addr = get_compile_time_arg_val(6);
    constexpr uint32_t sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t virtual_channel = get_compile_time_arg_val(8);

    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    experimental::Noc noc(noc_index);
    experimental::UnicastEndpoint unicast_endpoint;
    experimental::Semaphore semaphore(sem_id);
    constexpr experimental::AllocatorBankType bank_type = experimental::AllocatorBankType::DRAM;

    // Wait for semaphore to be set by the reader
    semaphore.down(1);

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            noc.async_write(
                unicast_endpoint,
                experimental::AllocatorBank<bank_type>(),
                bytes_per_transaction,
                {
                    .addr = local_l1_addr,
                },
                {
                    .bank_id = dram_channel,
                    .addr = dram_addr,
                },
                virtual_channel);
        }
        noc.async_write_barrier();
    }

    DeviceTimestampedData("Test id", test_id);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);

    DeviceTimestampedData("DRAM Channel", dram_channel);
    DeviceTimestampedData("Virtual Channel", virtual_channel);
}

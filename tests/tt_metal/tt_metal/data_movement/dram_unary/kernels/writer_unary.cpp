// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// L1 to DRAM write
void kernel_main() {
    constexpr uint32_t test_id = get_named_compile_time_arg_val("test_id");
    constexpr uint32_t num_of_transactions = get_named_compile_time_arg_val("num_transactions");
    constexpr uint32_t pages_per_transaction = get_named_compile_time_arg_val("pages_per_tx");
    constexpr uint32_t bytes_per_page = get_named_compile_time_arg_val("bytes_per_page");
    constexpr uint32_t dram_addr = get_named_compile_time_arg_val("dram_addr");
    constexpr uint32_t dram_channel = get_named_compile_time_arg_val("dram_channel");
    constexpr uint32_t local_l1_addr = get_named_compile_time_arg_val("l1_addr");
    constexpr uint32_t sem_id = get_named_compile_time_arg_val("sem_id");
    constexpr uint32_t virtual_channel = get_named_compile_time_arg_val("vc");

    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    constexpr bool dram = true;
    uint64_t dram_noc_addr = get_noc_addr_from_bank_id<dram>(dram_channel, dram_addr);

    uint32_t sem_addr = get_semaphore(sem_id);
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    // Wait for semaphore to be set by the reader
    noc_semaphore_wait(sem_ptr, 1);

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            noc_async_write(local_l1_addr, dram_noc_addr, bytes_per_transaction, noc_index, virtual_channel);
        }
        noc_async_write_barrier();
    }

    DeviceTimestampedData("Test id", test_id);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);

    DeviceTimestampedData("DRAM Channel", dram_channel);
    DeviceTimestampedData("Virtual Channel", virtual_channel);
}

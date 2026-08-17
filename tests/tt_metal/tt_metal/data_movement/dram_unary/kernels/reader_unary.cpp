// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DRAM to L1 read
void kernel_main() {
    constexpr uint32_t test_id = get_named_compile_time_arg_val("test_id");
    constexpr uint32_t num_of_transactions = get_named_compile_time_arg_val("num_transactions");
    constexpr uint32_t pages_per_transaction = get_named_compile_time_arg_val("pages_per_tx");
    constexpr uint32_t bytes_per_page = get_named_compile_time_arg_val("bytes_per_page");
    constexpr uint32_t dram_addr = get_named_compile_time_arg_val("dram_addr");
    constexpr uint32_t dram_channel = get_named_compile_time_arg_val("dram_channel");
    constexpr uint32_t local_l1_addr = get_named_compile_time_arg_val("l1_addr");
    constexpr uint32_t sem_id = get_named_compile_time_arg_val("sem_id");

    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    constexpr bool dram = true;
    uint64_t dram_noc_addr = get_noc_addr_from_bank_id<dram>(dram_channel, dram_addr);

    uint32_t sem_addr = get_semaphore(sem_id);
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            noc_async_read(dram_noc_addr, local_l1_addr, bytes_per_transaction);
        }
        noc_async_read_barrier();
    }

    // Set the semaphore to indicate that the writer can proceed
    noc_semaphore_set(sem_ptr, 1);

    DeviceTimestampedData("Test id", test_id);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
}

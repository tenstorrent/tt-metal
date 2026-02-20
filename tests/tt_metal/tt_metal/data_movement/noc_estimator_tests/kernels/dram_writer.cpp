// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DRAM write kernel for NOC estimator tests.
// Waits for semaphore from reader, then writes from local L1 to
// DRAM bank(s) in round-robin order.
// When num_banks=1, behaves as single-bank write.

#include "api/dataflow/dataflow_api.h"
#include "log_helpers.hpp"

void kernel_main() {
    constexpr uint32_t local_l1_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dram_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(2);
    constexpr uint32_t bytes_per_transaction = get_compile_time_arg_val(3);
    constexpr uint32_t test_id = get_compile_time_arg_val(4);
    constexpr uint32_t sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t num_banks = get_compile_time_arg_val(6);

    // Metadata args
    constexpr uint32_t memory_type = get_compile_time_arg_val(7);
    constexpr uint32_t mechanism = get_compile_time_arg_val(8);
    constexpr uint32_t pattern = get_compile_time_arg_val(9);

    constexpr bool dram = true;

    uint32_t sem_addr = get_semaphore(sem_id);
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    noc_semaphore_wait(sem_ptr, 1);

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            uint32_t bank = i % num_banks;
            uint64_t dram_noc_addr = get_noc_addr_from_bank_id<dram>(bank, dram_addr);
            noc_async_write(local_l1_addr, dram_noc_addr, bytes_per_transaction);
        }
        noc_async_write_barrier();
    }

    log_estimator_metadata(
        test_id, noc_index, num_of_transactions, bytes_per_transaction, memory_type, mechanism, pattern, 0, 0, 0, 0);
}

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DRAM read kernel for NOC estimator tests.
// Reads from a DRAM bank to local L1. Sets a semaphore when done
// to signal the DRAM writer kernel.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "log_helpers.hpp"

void kernel_main() {
    constexpr uint32_t local_l1_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dram_addr = get_compile_time_arg_val(1);
    constexpr uint32_t dram_channel = get_compile_time_arg_val(2);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(3);
    constexpr uint32_t bytes_per_transaction = get_compile_time_arg_val(4);
    constexpr uint32_t test_id = get_compile_time_arg_val(5);
    constexpr uint32_t sem_id = get_compile_time_arg_val(6);

    // Metadata args
    constexpr uint32_t memory_type = get_compile_time_arg_val(7);
    constexpr uint32_t mechanism = get_compile_time_arg_val(8);
    constexpr uint32_t pattern = get_compile_time_arg_val(9);

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

    // Signal the writer kernel that data is ready in L1
    noc_semaphore_set(sem_ptr, 1);

    log_estimator_metadata(
        test_id, noc_index, num_of_transactions, bytes_per_transaction, memory_type, mechanism, pattern, 0, 0, 0, 0);
}

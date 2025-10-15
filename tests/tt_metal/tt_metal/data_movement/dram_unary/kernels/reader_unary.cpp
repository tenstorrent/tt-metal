// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

// DRAM to L1 read
void kernel_main() {
    constexpr uint32_t test_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t pages_per_transaction = get_compile_time_arg_val(2);
    constexpr uint32_t bytes_per_page = get_compile_time_arg_val(3);
    constexpr uint32_t dram_addr = get_compile_time_arg_val(4);
    constexpr uint32_t dram_channel = get_compile_time_arg_val(5);
    constexpr uint32_t local_l1_addr = get_compile_time_arg_val(6);
    constexpr uint32_t sem_id = get_compile_time_arg_val(7);

    // Debug prints for compile time arguments
    DPRINT << "=== Reader Kernel Debug Info ===" << ENDL();
    DPRINT << "test_id: " << test_id << ENDL();
    DPRINT << "num_of_transactions: " << num_of_transactions << ENDL();
    DPRINT << "pages_per_transaction: " << pages_per_transaction << ENDL();
    DPRINT << "bytes_per_page: " << bytes_per_page << ENDL();
    DPRINT << "dram_addr: " << HEX() << dram_addr << DEC() << ENDL();
    DPRINT << "dram_channel: " << dram_channel << ENDL();
    DPRINT << "local_l1_addr: " << HEX() << local_l1_addr << DEC() << ENDL();
    DPRINT << "sem_id: " << sem_id << ENDL();

    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    constexpr bool dram = true;
    uint64_t dram_noc_addr = get_noc_addr_from_bank_id<dram>(dram_channel, dram_addr);

    uint32_t sem_addr = get_semaphore(sem_id);
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    DPRINT << "bytes_per_transaction: " << bytes_per_transaction << ENDL();
    DPRINT << "dram_noc_addr: " << HEX() << dram_noc_addr << DEC() << ENDL();
    DPRINT << "sem_addr: " << HEX() << sem_addr << DEC() << ENDL();
    DPRINT << "Waiting for semaphore..." << ENDL();

    // Wait for semaphore to be set by the writer
    noc_semaphore_wait(sem_ptr, 1);

    DPRINT << "Semaphore received, starting NOC reads..." << ENDL();

    uint64_t tx_start;
    uint64_t tx_end;
    {
        DeviceZoneScopedN("RISCV1");
        tx_start = get_timestamp();
        uint64_t curr_dram_noc_addr = dram_noc_addr;
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            // DPRINT << "Transaction " << i << "/" << num_of_transactions << ": reading " << bytes_per_transaction << "
            // bytes" << ENDL();
            noc_async_read(curr_dram_noc_addr, local_l1_addr, bytes_per_transaction);
            curr_dram_noc_addr += bytes_per_transaction;
        }
        // DPRINT << "All reads issued, waiting for barrier..." << ENDL();
        noc_async_read_barrier();
    }
    tx_end = get_timestamp();
    uint64_t tx_diff = tx_end - tx_start;
    DPRINT << "Transaction time reads: " << tx_diff << ENDL();
    uint64_t bw_gbs = bytes_per_transaction * num_of_transactions / tx_diff * 1.35;
    DPRINT << "Transaction bandwidth: " << bw_gbs << " GB/s" << ENDL();

    DeviceTimestampedData("Test id", test_id);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
}

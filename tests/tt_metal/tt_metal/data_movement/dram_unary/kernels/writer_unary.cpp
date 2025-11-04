// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

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

    // Debug prints for compile time arguments
    DPRINT << "=== Writer Kernel Debug Info ===" << ENDL();
    DPRINT << "test_id: " << test_id << ENDL();
    DPRINT << "num_of_transactions: " << num_of_transactions << ENDL();
    DPRINT << "pages_per_transaction: " << pages_per_transaction << ENDL();
    DPRINT << "bytes_per_page: " << bytes_per_page << ENDL();
    DPRINT << "dram_addr: " << HEX() << dram_addr << DEC() << ENDL();
    DPRINT << "dram_channel: " << dram_channel << ENDL();
    DPRINT << "local_l1_addr: " << HEX() << local_l1_addr << DEC() << ENDL();
    DPRINT << "sem_id: " << sem_id << ENDL();
    DPRINT << "virtual_channel: " << virtual_channel << ENDL();

    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    constexpr bool dram = true;

    // Get the base NOC address for the DRAM bank
    uint64_t dram_base_noc_addr = get_noc_addr_from_bank_id<dram>(dram_channel, 0);

    // Add the large offset directly to the NOC address
    uint64_t dram_offset = dram_addr;  // 0x2000000000 + bytes_per_transaction;
    uint64_t dram_noc_addr = dram_base_noc_addr + dram_offset;

    DPRINT << "dram_base_noc_addr: " << HEX() << dram_base_noc_addr << DEC() << ENDL();
    DPRINT << "dram_offset: " << HEX() << dram_offset << DEC() << ENDL();
    DPRINT << "dram_noc_addr: " << HEX() << dram_noc_addr << DEC() << ENDL();
    DPRINT << "DRAM top part: " << HEX() << get_noc_addr_from_bank_id<dram>(dram_channel, 0) << DEC() << ENDL();
    uint32_t sem_addr = get_semaphore(sem_id);
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    DPRINT << "bytes_per_transaction: " << bytes_per_transaction << ENDL();
    DPRINT << "dram_noc_addr: " << HEX() << dram_noc_addr << DEC() << ENDL();
    DPRINT << "sem_addr: " << HEX() << sem_addr << DEC() << ENDL();
    DPRINT << "Starting NOC writes..." << ENDL();

    uint64_t tx_start;
    uint64_t tx_end;

    // uint64_t axi_disabled_addr = 0x100FFB20100;
    // uint64_t axi_disabled_addr_noc = get_noc_addr_from_bank_id<dram>(dram_channel, axi_disabled_addr);
    // uint32_t axi_disabled_data = 0x00006005;
    // volatile tt_l1_ptr std::uint32_t* axi_data_ptr = (volatile tt_l1_ptr uint32_t*)(0x100000);
    // *axi_data_ptr = axi_disabled_data;

    uint64_t mc_base5_addr = 0x100FC109400;
    uint64_t mc_base5_addr_noc = get_noc_addr_from_bank_id<dram>(dram_channel, mc_base5_addr);
    uint32_t mc_base5_data = 0x3;
    volatile tt_l1_ptr std::uint32_t* mc_base5_data_ptr = (volatile tt_l1_ptr uint32_t*)(0x100000);
    *mc_base5_data_ptr = mc_base5_data;
    noc_async_write(0x100000, mc_base5_addr_noc, 4);
    noc_async_write_barrier();

    uint64_t mc_base2_addr = 0x100FC1046A8;
    uint64_t mc_base2_addr_noc = get_noc_addr_from_bank_id<dram>(dram_channel, mc_base2_addr);
    uint32_t mc_base2_data = 0x1;
    volatile tt_l1_ptr std::uint32_t* mc_base2_data_ptr = (volatile tt_l1_ptr uint32_t*)(0x100000);
    *mc_base2_data_ptr = mc_base2_data;
    noc_async_write(0x100000, mc_base2_addr_noc, 4);
    noc_async_write_barrier();

    {
        DeviceZoneScopedN("RISCV0");
        tx_start = get_timestamp();
        uint64_t curr_dram_noc_addr = dram_noc_addr;

        for (uint32_t i = 0; i < num_of_transactions; i++) {
            // DPRINT << "Transaction " << i << "/" << num_of_transactions << ": writing " << bytes_per_transaction << "
            // bytes" << ENDL();
            noc_async_write(local_l1_addr, curr_dram_noc_addr, bytes_per_transaction, noc_index, virtual_channel);
            // DPRINT << "Transaction " << i << "/" << num_of_transactions << ": wrote " << bytes_per_transaction
            //        << " bytes" << ENDL();
            curr_dram_noc_addr += bytes_per_transaction;
        }
        // DPRINT << "All writes issued, waiting for barrier..." << ENDL();
        noc_async_write_barrier();
        tx_end = get_timestamp();
        // DPRINT << "Write barrier complete!" << ENDL();
    }
    uint64_t tx_diff = tx_end - tx_start;
    DPRINT << "Transaction time writes: " << tx_diff << ENDL();
    uint64_t bw_gbs = bytes_per_transaction * num_of_transactions / tx_diff * 1.35;
    DPRINT << "Transaction bandwidth write: " << bw_gbs << " GB/s" << ENDL();

    // Set the semaphore to indicate that the reader can proceed
    DPRINT << "Setting semaphore to signal reader..." << ENDL();
    noc_semaphore_set(sem_ptr, 1);
    DPRINT << "Semaphore set, writer complete!" << ENDL();

    DeviceTimestampedData("Test id", test_id);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);

    DeviceTimestampedData("DRAM Channel", dram_channel);
    DeviceTimestampedData("Virtual Channel", virtual_channel);
}

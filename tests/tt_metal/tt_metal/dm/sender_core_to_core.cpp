// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

// L1 to L1 send
void kernel_main() {
    constexpr uint32_t src_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dst_addr = get_compile_time_arg_val(1);
    // constexpr uint32_t num_of_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t transaction_num_pages = get_compile_time_arg_val(2);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(3);
    // constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(4);
    // constexpr uint32_t test_id = get_compile_time_arg_val(5);

    auto sem = get_semaphore(get_arg_val<uint32_t>(0));
    DPRINT << "Get sem result: " << sem << ENDL();

    // auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(0)));
    uint64_t sem_addr = get_noc_addr(2, 3, sem);

    constexpr uint32_t transaction_size_bytes = transaction_num_pages * page_size_bytes;
    // constexpr uint32_t total_num_pages = num_of_transactions * transaction_num_pages;

    // DeviceTimestampedData("Number of transactions", num_of_transactions);
    // DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    // DeviceTimestampedData("Test id", test_id);

    {
        // DeviceZoneScopedN("SENDER");
        uint64_t dst_noc_addr = get_noc_addr(1, 1, dst_addr);  // TODO: Sending to right location?
        // uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

        noc_async_write(src_addr, dst_noc_addr, transaction_size_bytes);
        noc_async_write_barrier();

        DPRINT << "Sender before sem call" << ENDL();
        noc_semaphore_inc(sem_addr, 1);
        DPRINT << "Sender after sem call" << ENDL();
    }
}

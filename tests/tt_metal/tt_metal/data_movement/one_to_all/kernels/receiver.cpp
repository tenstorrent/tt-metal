// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

// L1 to L1 receive
void kernel_main() {
    constexpr uint32_t src_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dst_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(2);
    constexpr uint32_t transaction_num_pages = get_compile_time_arg_val(3);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t test_id = get_compile_time_arg_val(5);

    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(0)));

    constexpr uint32_t transaction_size_bytes = transaction_num_pages * page_size_bytes;

    {
        DeviceZoneScopedN("RISCV1");
        noc_semaphore_wait(sem_ptr, 1);
    }

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);
}

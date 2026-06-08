// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Receiver semaphore kernel
void kernel_main() {
    constexpr uint32_t num_of_transactions = get_named_compile_time_arg_val("num_transactions");
    constexpr uint32_t pages_per_transaction = get_named_compile_time_arg_val("pages_per_tx");
    constexpr uint32_t bytes_per_page = get_named_compile_time_arg_val("bytes_per_page");
    constexpr uint32_t test_id = get_named_compile_time_arg_val("test_id");
    constexpr uint32_t sender_sem_id = get_named_compile_time_arg_val("sender_sem_id");
    constexpr uint32_t receiver_sem_id = get_named_compile_time_arg_val("receiver_sem_id");
    constexpr uint32_t sender_core_coordinates = get_named_compile_time_arg_val("sender_coords");

    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    auto receiver_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);

    uint64_t sender_sem_noc_addr =
        get_noc_addr(sender_core_coordinates >> 16, sender_core_coordinates & 0xFFFF, sender_sem_addr);
    {
        DeviceZoneScopedN("RISCV1");

        for (uint32_t i = 0; i < num_of_transactions; i++) {
            // Set the receiver's semaphore to 0 to indicate that it is ready to receive the data
            noc_semaphore_set(receiver_sem_ptr, 0);

            // Increment the sender's semaphore
            noc_semaphore_inc(sender_sem_noc_addr, 1);

            // Wait for semaphore to be set by the sender
            noc_semaphore_wait(receiver_sem_ptr, 1);
        }
    }

    DeviceTimestampedData("Test id", test_id);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
}

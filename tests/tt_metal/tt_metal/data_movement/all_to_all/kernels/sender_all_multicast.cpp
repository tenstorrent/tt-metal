/*
TO-DO:
    - "All sender multicast" kernel
    - Multicast write from each passed-in NOC coordinates (for now this will probably end up being every core within the
range)
    - Update a semaphore per receiver core
    - QUESTION: Do we want to increment the semaphore once at the end (for all transactions combined) or once per
transaction so that the stored value is equal to the number of sender cores?
*/

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

// L1 to L1 send
void kernel_main() {
    /* Initializing compile-time arguments */

    constexpr uint32_t test_id = get_compile_time_arg_val(0);

    constexpr uint32_t src_addr = get_compile_time_arg_val(1);
    constexpr uint32_t dst_addr = get_compile_time_arg_val(2);

    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(3);
    constexpr uint32_t transaction_num_pages = get_compile_time_arg_val(4);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t total_transaction_size_bytes = get_compile_time_arg_val(6);

    constexpr bool linked = get_compile_time_arg_val(7);
    constexpr uint32_t semaphore_id = get_compile_time_arg_val(8);

    constexpr uint32_t num_masters = get_compile_time_arg_val(9);
    constexpr uint32_t num_subordinates = get_compile_time_arg_val(10);

    constexpr uint32_t mst_coords_offset = get_compile_time_arg_val(11);
    constexpr uint32_t sub_coords_offset = get_compile_time_arg_val(12);

    constexpr uint32_t sub_start_x = get_compile_time_arg_val(13);
    constexpr uint32_t sub_start_y = get_compile_time_arg_val(14);
    constexpr uint32_t sub_end_x = get_compile_time_arg_val(15);
    constexpr uint32_t sub_end_y = get_compile_time_arg_val(16);

    /* Initializing derived arguments */

    uint32_t semaphore = get_semaphore(semaphore_id);
    uint64_t dst_noc_addr_multicast = get_noc_multicast_addr(
        sub_start_x,
        sub_start_y,
        sub_end_x,
        sub_end_y,
        dst_addr);  // NOTE: This needs to be added to the API // QUESTION: Shouldn't we also put in the NOC ID for
                    // this?
    constexpr uint32_t transaction_size_bytes = transaction_num_pages * page_size_bytes;

    /* Running the Sender Kernel */
    {
        DeviceZoneScopedN("RISCV0");
        // For all master cores
        for (uint32_t master = 0; master < num_masters; master++) {
            // Obtain the NOC address of the current master core
            uint32_t mst_x = get_compile_time_arg_val(mst_coords_offset + 2 * master);
            uint32_t mst_y = get_compile_time_arg_val(mst_coords_offset + 2 * master + 1);
            uint64_t src_noc_addr = get_noc_addr(mst_x, mst_y, src_addr);

            // Send the multicast write transactions
            for (uint32_t i = 0; i < num_of_transactions - 1; i++) {
                noc_async_write_multicast_loopback_src(
                    src_noc_addr, dst_noc_addr_multicast, transaction_size_bytes, num_subordinates, linked);
            }
            // Last packet is sent separately to unlink the transaction, so the next one can use the VC and do its own
            // path reservation
            noc_async_write_multicast_loopback_src(
                src_noc_addr, dst_noc_addr_multicast, transaction_size_bytes, num_subordinates, false);
            noc_async_write_barrier();

            // Increment semaphore for each receiver core. This should result in all semaphores having a value equal to
            // the number of sender cores
            for (uint32_t subordinate = 0; subordinate < num_subordinates; subordinate++) {
                uint32_t sub_x = get_compile_time_arg_val(sub_coords_offset + 2 * subordinate);
                uint32_t sub_y = get_compile_time_arg_val(sub_coords_offset + 2 * subordinate + 1);
                uint64_t sem_addr = get_noc_addr(sub_x, sub_y, semaphore);
                noc_semaphore_inc(sem_addr, 1);
            }
        }
    }

    // NOTE: This part will definitely require updating. Printing more meaningful information
    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);
}

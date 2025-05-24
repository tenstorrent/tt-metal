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
    // Initializing compile-time arguments
    uint32_t src_addr = get_compile_time_arg_val(0);
    uint32_t dst_addr = get_compile_time_arg_val(1);
    constexpr uint32_t test_id = get_compile_time_arg_val(2);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(3);
    constexpr uint32_t transaction_num_pages = get_compile_time_arg_val(4);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t total_transaction_size_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t num_masters = get_compile_time_arg_val(7);
    constexpr uint32_t num_subordinates = get_compile_time_arg_val(8);
    constexpr bool linked = get_compile_time_arg_val(9);
    constexpr uint32_t semaphore_id = get_compile_time_arg_val(10);

    /*
    uint32_t sub_start_x = get_arg_val<uint32_t>(1);
    uint32_t sub_start_y = get_arg_val<uint32_t>(2);
    uint32_t sub_end_x = get_arg_val<uint32_t>(3);
    uint32_t sub_end_y = get_arg_val<uint32_t>(4);*/

    uint32_t semaphore = get_semaphore(semaphore_id);

    // For all master cores
    // for i = 0, i < num_src_addressses, i++
    for (uint32_t master = 0; master < num_masters; master++) {
        // src_addr = get_compile_time_arg_val(0) + i * transaction_size_bytes;
        // Perform multicast write to all receiver cores (use multicast loopback src)
        // For now, just assume the total number of cores is divisble by 4.

        constexpr uint32_t transaction_size_bytes = transaction_num_pages * page_size_bytes;
        uint64_t dst_noc_addr_multicast =
            get_noc_multicast_addr(sub_start_x, sub_start_y, sub_end_x, sub_end_y, dst_addr);

        {
            DeviceZoneScopedN("RISCV0");
            for (uint32_t i = 0; i < num_of_transactions - 1; i++) {
                noc_async_write_multicast_loopback_src(
                    src_addr, dst_noc_addr_multicast, transaction_size_bytes, num_subordinates, linked);
            }
            // Last packet is sent separately to unlink the transaction, so the next one can use the VC and do its own
            // path reservation
            noc_async_write_multicast_loopback_src(
                src_addr, dst_noc_addr_multicast, transaction_size_bytes, num_subordinates, false);
            noc_async_write_barrier();
        }

        // Increment semaphore for each receiver core. This should result in all semaphores having a value equal to the
        // number of sender cores

        // BELOW is from one to all sender_multicast.cpp

        for (uint32_t subordinate = 0; subordinate < num_subordinates; subordinate++) {
            for (auto& core : corerange_to_cores(subordinate_core_set)) {
                if (!test_config.loopback &&
                    (core.x == test_config.master_core_coord.x && core.y == test_config.master_core_coord.y)) {
                    continue;
                }
                CoreCoord worker = device->worker_core_from_logical_core(core);  // This part namely
                master_run_args.push_back(worker.x);
                master_run_args.push_back(worker.y);
            }

            uint32_t dest_x = get_arg_val<uint32_t>(subordinate_coords_offset + 2 * subordinate);
            uint32_t dest_y = get_arg_val<uint32_t>(subordinate_coords_offset + 2 * subordinate + 1);

            uint64_t sem_addr = get_noc_addr(dest_x, dest_y, semaphore);
            noc_semaphore_inc(sem_addr, 1);
        }

        DeviceTimestampedData("Number of transactions", num_of_transactions);
        DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
        DeviceTimestampedData("Test id", test_id);
    }

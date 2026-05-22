// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"

// Sender semaphore kernel (device 2.0 API).
// Mirrors sender_multicast_sem.cpp but uses Noc::async_write_multicast for data and
// the new Semaphore<>::set_multicast(dst_sem, ...) overload to signal receivers at a
// different sem L1 offset than the sender's local valid_sem.
void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t mst_base_addr = get_compile_time_arg_val(0);
    constexpr uint32_t sub_base_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_transaction = get_compile_time_arg_val(3);
    constexpr uint32_t bytes_per_page = get_compile_time_arg_val(4);
    constexpr uint32_t test_id = get_compile_time_arg_val(5);
    constexpr uint32_t num_subordinates = get_compile_time_arg_val(6);
    constexpr bool is_linked = get_compile_time_arg_val(7);
    constexpr bool loopback = get_compile_time_arg_val(8);
    uint32_t start_x = get_compile_time_arg_val(9);
    uint32_t start_y = get_compile_time_arg_val(10);
    uint32_t end_x = get_compile_time_arg_val(11);
    uint32_t end_y = get_compile_time_arg_val(12);

    // Specific for Multicast Schemes
    constexpr uint32_t multicast_scheme_type = get_compile_time_arg_val(13);
    constexpr uint32_t sub_grid_size_x = get_compile_time_arg_val(14);
    constexpr uint32_t sub_grid_size_y = get_compile_time_arg_val(15);

    // Semaphore arguments
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(16);
    constexpr uint32_t sender_valid_sem_id = get_compile_time_arg_val(17);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(18);

    // Derivative values
    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;
    constexpr uint32_t bytes = bytes_per_transaction * num_of_transactions;

    if (noc_index == 1) {
        std::swap(start_x, end_x);
        std::swap(start_y, end_y);
    }

    Noc noc(noc_index);
    UnicastEndpoint unicast_endpoint;
    MulticastEndpoint multicast_endpoint;

    Semaphore<> sender_sem(sender_sem_id);
    Semaphore<> sender_valid_sem(sender_valid_sem_id);
    Semaphore<> receiver_sem(receiver_sem_id);

    constexpr Noc::McastMode mcast_mode = loopback ? Noc::McastMode::INCLUDE_SRC : Noc::McastMode::EXCLUDE_SRC;

    {
        DeviceZoneScopedN("RISCV0");

        for (uint32_t i = 0; i < num_of_transactions - 1; i++) {
            sender_sem.wait(num_subordinates);
            sender_sem.set(0);

            noc.async_write_multicast<mcast_mode>(
                unicast_endpoint,
                multicast_endpoint,
                bytes_per_transaction,
                num_subordinates,
                {.addr = mst_base_addr},
                {.noc_x_start = start_x,
                 .noc_y_start = start_y,
                 .noc_x_end = end_x,
                 .noc_y_end = end_y,
                 .addr = sub_base_addr},
                is_linked);

            sender_valid_sem.template set_multicast<mcast_mode>(
                noc, receiver_sem, start_x, start_y, end_x, end_y, num_subordinates, is_linked);
        }

        // Last transaction must unlink so the next workload can reserve its own path on the VC.
        sender_sem.wait(num_subordinates);
        sender_sem.set(0);

        noc.async_write_multicast<mcast_mode>(
            unicast_endpoint,
            multicast_endpoint,
            bytes_per_transaction,
            num_subordinates,
            {.addr = mst_base_addr},
            {.noc_x_start = start_x,
             .noc_y_start = start_y,
             .noc_x_end = end_x,
             .noc_y_end = end_y,
             .addr = sub_base_addr});

        sender_valid_sem.template set_multicast<mcast_mode>(
            noc, receiver_sem, start_x, start_y, end_x, end_y, num_subordinates, false);
    }

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("NoC Index", noc_index);
    if constexpr (multicast_scheme_type != 0) {
        DeviceTimestampedData("Multicast Scheme Type", multicast_scheme_type);
        DeviceTimestampedData("Subordinate Grid Size X", sub_grid_size_x);
        DeviceTimestampedData("Subordinate Grid Size Y", sub_grid_size_y);
    }
    DeviceTimestampedData("Number of subordinates", num_subordinates);
}

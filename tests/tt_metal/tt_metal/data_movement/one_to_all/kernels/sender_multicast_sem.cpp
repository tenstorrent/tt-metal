// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Sender semaphore kernel
void kernel_main() {
    uint32_t mst_base_addr = get_named_compile_time_arg_val("mst_base_addr");
    uint32_t sub_base_addr = get_named_compile_time_arg_val("sub_base_addr");
    constexpr uint32_t num_of_transactions = get_named_compile_time_arg_val("num_transactions");
    constexpr uint32_t pages_per_transaction = get_named_compile_time_arg_val("pages_per_tx");
    constexpr uint32_t bytes_per_page = get_named_compile_time_arg_val("bytes_per_page");
    constexpr uint32_t test_id = get_named_compile_time_arg_val("test_id");
    constexpr uint32_t num_subordinates = get_named_compile_time_arg_val("num_subordinates");
    constexpr bool is_linked = get_named_compile_time_arg_val("is_linked");
    constexpr bool loopback = get_named_compile_time_arg_val("loopback");
    constexpr uint32_t start_x = get_named_compile_time_arg_val("start_x");
    constexpr uint32_t start_y = get_named_compile_time_arg_val("start_y");
    constexpr uint32_t end_x = get_named_compile_time_arg_val("end_x");
    constexpr uint32_t end_y = get_named_compile_time_arg_val("end_y");
    constexpr uint32_t multicast_scheme_type = get_named_compile_time_arg_val("mcast_scheme_type");
    constexpr uint32_t sub_grid_size_x = get_named_compile_time_arg_val("sub_grid_size_x");
    constexpr uint32_t sub_grid_size_y = get_named_compile_time_arg_val("sub_grid_size_y");
    constexpr uint32_t sender_sem_id = get_named_compile_time_arg_val("sender_sem_id");
    constexpr uint32_t sender_valid_sem_id = get_named_compile_time_arg_val("sender_valid_sem_id");
    constexpr uint32_t receiver_sem_id = get_named_compile_time_arg_val("receiver_sem_id");

    // Derivative values
    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    uint64_t dst_noc_addr_multicast = noc_index == 0
                                          ? get_noc_multicast_addr(start_x, start_y, end_x, end_y, sub_base_addr)
                                          : get_noc_multicast_addr(end_x, end_y, start_x, start_y, sub_base_addr);

    uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    auto sender_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);

    uint32_t sender_valid_sem_addr = get_semaphore(sender_valid_sem_id);

    uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    uint64_t dst_sem_noc_addr_multicast =
        noc_index == 0 ? get_noc_multicast_addr(start_x, start_y, end_x, end_y, receiver_sem_addr)
                       : get_noc_multicast_addr(end_x, end_y, start_x, start_y, receiver_sem_addr);

    {
        for (uint32_t i = 0; i < num_of_transactions - 1; i++) {
            // Wait for semaphore to be set by the receiver
            noc_semaphore_wait(sender_sem_ptr, num_subordinates);
            noc_semaphore_set(sender_sem_ptr, 0);

            if constexpr (loopback) {
                noc_async_write_multicast_loopback_src(
                    mst_base_addr, dst_noc_addr_multicast, bytes_per_transaction, num_subordinates, is_linked);
                noc_semaphore_set_multicast_loopback_src(
                    sender_valid_sem_addr, dst_sem_noc_addr_multicast, num_subordinates, is_linked);
            } else {
                noc_async_write_multicast(
                    mst_base_addr, dst_noc_addr_multicast, bytes_per_transaction, num_subordinates, is_linked);
                noc_semaphore_set_multicast(
                    sender_valid_sem_addr, dst_sem_noc_addr_multicast, num_subordinates, is_linked);
            }
        }

        // Wait for semaphore to be set by the receiver
        noc_semaphore_wait(sender_sem_ptr, num_subordinates);
        noc_semaphore_set(sender_sem_ptr, 0);

        // Last packet is sent separately to unlink the transaction,
        // so the next one can use the VC and do its own path reservation
        if constexpr (loopback) {
            noc_async_write_multicast_loopback_src(
                mst_base_addr, dst_noc_addr_multicast, bytes_per_transaction, num_subordinates, false);
            noc_semaphore_set_multicast_loopback_src(
                sender_valid_sem_addr, dst_sem_noc_addr_multicast, num_subordinates, false);
        } else {
            noc_async_write_multicast(
                mst_base_addr, dst_noc_addr_multicast, bytes_per_transaction, num_subordinates, false);
            noc_semaphore_set_multicast(sender_valid_sem_addr, dst_sem_noc_addr_multicast, num_subordinates, false);
        }
    }
}

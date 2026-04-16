// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "barrier_sync.hpp"

void kernel_main() {
    constexpr uint32_t test_id = get_compile_time_arg_val(0);
    constexpr uint32_t physical_start_x = get_compile_time_arg_val(1);
    constexpr uint32_t physical_start_y = get_compile_time_arg_val(2);
    constexpr uint32_t physical_end_x = get_compile_time_arg_val(3);
    constexpr uint32_t physical_end_y = get_compile_time_arg_val(4);
    constexpr uint32_t num_cores_c_dim = get_compile_time_arg_val(5);
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t sender_valid_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(8);

    uint32_t l1_base_address = get_arg_val<uint32_t>(0);
    uint32_t num_k_subblocks_this_core = get_arg_val<uint32_t>(1);
    uint32_t k_subblock_size_bytes = get_arg_val<uint32_t>(2);
    uint32_t in0_mcast_output_addr = get_arg_val<uint32_t>(3);
    uint32_t my_col_idx = get_arg_val<uint32_t>(4);
    uint32_t num_subblocks_k_dim = get_arg_val<uint32_t>(5);
    // Barrier synchronization args
    uint32_t barrier_sem_id = get_arg_val<uint32_t>(6);
    uint32_t barrier_coord_x = get_arg_val<uint32_t>(7);
    uint32_t barrier_coord_y = get_arg_val<uint32_t>(8);
    uint32_t num_cores = get_arg_val<uint32_t>(9);
    uint32_t local_barrier_addr = get_arg_val<uint32_t>(10);
    uint32_t barrier_done_sem_id = get_arg_val<uint32_t>(11);
    // col_phys_x[c] = get_arg_val<uint32_t>(12 + c) for c in [0, num_cores_c_dim)

    uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    uint32_t sender_valid_sem_addr = get_semaphore(sender_valid_sem_id);
    uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);

    volatile tt_l1_ptr uint32_t* sender_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);

    // Multicast addresses for my row (same Y for all cores in the row, varying X)
    uint64_t row_mcast_base = get_noc_multicast_addr(physical_start_x, my_y[0], physical_end_x, my_y[0], 0);

    barrier_sync(
        barrier_sem_id,
        barrier_done_sem_id,
        barrier_coord_x,
        barrier_coord_y,
        num_cores,
        local_barrier_addr,
        physical_start_x,
        physical_start_y,
        physical_end_x,
        physical_end_y);

    // K-loop: iterate through all K subblocks, rotating sender across columns
    uint32_t local_k_send_idx = 0;

    for (uint32_t k = 0; k < num_subblocks_k_dim; k++) {
        uint32_t sender_col = k % num_cores_c_dim;
        uint32_t output_addr = in0_mcast_output_addr + k * k_subblock_size_bytes;

        if (my_col_idx == sender_col) {
            // --- SENDER: multicast this K subblock to all cores in my row ---
            uint32_t src_addr = l1_base_address + local_k_send_idx * k_subblock_size_bytes;
            local_k_send_idx++;

            // Wait for all receivers in the row to signal readiness
            noc_semaphore_wait(sender_sem_ptr, num_cores_c_dim - 1);
            noc_semaphore_set(sender_sem_ptr, 0);

            if constexpr (num_cores_c_dim > 1) {
                uint64_t dst_data_mcast_addr = row_mcast_base | output_addr;
                noc_async_write_multicast_loopback_src(
                    src_addr, dst_data_mcast_addr, k_subblock_size_bytes, num_cores_c_dim, true);

                uint64_t dst_receiver_sem_mcast_addr = row_mcast_base | receiver_sem_addr;
                noc_semaphore_set_multicast_loopback_src(
                    sender_valid_sem_addr, dst_receiver_sem_mcast_addr, num_cores_c_dim, false);
            } else {
                // Single column: unicast self-write (HW limitation with single-core multicast loopback)
                uint64_t local_dest_addr = get_noc_addr(my_x[0], my_y[0], output_addr);
                noc_async_write(src_addr, local_dest_addr, k_subblock_size_bytes);
                noc_async_write_barrier();
                noc_semaphore_set(receiver_sem_ptr, 1);
            }
        } else {
            // --- RECEIVER: signal readiness to sender ---
            uint32_t sender_phys_x = get_arg_val<uint32_t>(12 + sender_col);
            uint64_t sender_sem_noc_addr = get_noc_addr(sender_phys_x, my_y[0], sender_sem_addr);
            noc_semaphore_inc(sender_sem_noc_addr, 1);
        }

        // All cores wait for data arrival, then reset for next iteration
        noc_semaphore_wait(receiver_sem_ptr, 1);
        noc_semaphore_set(receiver_sem_ptr, 0);
    }

    DeviceTimestampedData("Test id", test_id);
}

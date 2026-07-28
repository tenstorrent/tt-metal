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
    constexpr uint32_t num_cores_r_dim = get_compile_time_arg_val(5);
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t sender_valid_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(8);

    uint32_t l1_base_address = get_arg_val<uint32_t>(0);
    uint32_t num_k_subblocks_this_core = get_arg_val<uint32_t>(1);
    uint32_t k_subblock_size_bytes = get_arg_val<uint32_t>(2);
    uint32_t in1_mcast_output_addr = get_arg_val<uint32_t>(3);
    uint32_t my_row_idx = get_arg_val<uint32_t>(4);
    uint32_t num_subblocks_k_dim = get_arg_val<uint32_t>(5);
    // Barrier synchronization args
    uint32_t barrier_sem_id = get_arg_val<uint32_t>(6);
    uint32_t barrier_coord_x = get_arg_val<uint32_t>(7);
    uint32_t barrier_coord_y = get_arg_val<uint32_t>(8);
    uint32_t num_cores = get_arg_val<uint32_t>(9);
    uint32_t local_barrier_addr = get_arg_val<uint32_t>(10);
    uint32_t barrier_done_sem_id = get_arg_val<uint32_t>(11);
    // row_phys_y[r] at runtime arg index (12 + r)

    uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    uint32_t sender_valid_sem_addr = get_semaphore(sender_valid_sem_id);
    uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);

    volatile tt_l1_ptr uint32_t* sender_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);

    // NOC1 has reversed routing, so swap start_y and end_y for the column multicast.
    uint64_t col_mcast_base = get_noc_multicast_addr(my_x[0], physical_end_y, my_x[0], physical_start_y, 0);

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

    // Outside the timer zone so the stamp doesn't inflate duration_cycles.
    uint32_t local_k_send_idx = 0;
    {
        DeviceZoneScopedN("RISCV1");

        for (uint32_t k = 0; k < num_subblocks_k_dim; k++) {
            uint32_t sender_row = k % num_cores_r_dim;
            uint32_t output_addr = in1_mcast_output_addr + k * k_subblock_size_bytes;

            if (my_row_idx == sender_row) {
                uint32_t src_addr = l1_base_address + local_k_send_idx * k_subblock_size_bytes;
                local_k_send_idx++;

                noc_semaphore_wait(sender_sem_ptr, num_cores_r_dim - 1);
                noc_semaphore_set(sender_sem_ptr, 0);

                if constexpr (num_cores_r_dim > 1) {
                    uint64_t dst_data_mcast_addr = col_mcast_base | output_addr;
                    noc_async_write_multicast_loopback_src(
                        src_addr, dst_data_mcast_addr, k_subblock_size_bytes, num_cores_r_dim, true);

                    uint64_t dst_receiver_sem_mcast_addr = col_mcast_base | receiver_sem_addr;
                    noc_semaphore_set_multicast_loopback_src(
                        sender_valid_sem_addr, dst_receiver_sem_mcast_addr, num_cores_r_dim, false);
                } else {
                    // R=1: unicast self-write (HW limit on single-core multicast loopback).
                    uint64_t local_dest_addr = get_noc_addr(my_x[0], my_y[0], output_addr);
                    noc_async_write(src_addr, local_dest_addr, k_subblock_size_bytes);
                    noc_async_write_barrier();
                    noc_semaphore_set(receiver_sem_ptr, 1);
                }
            } else {
                uint32_t sender_phys_y = get_arg_val<uint32_t>(12 + sender_row);
                uint64_t sender_sem_noc_addr = get_noc_addr(my_x[0], sender_phys_y, sender_sem_addr);
                noc_semaphore_inc(sender_sem_noc_addr, 1);
            }

            noc_semaphore_wait(receiver_sem_ptr, 1);
            noc_semaphore_set(receiver_sem_ptr, 0);
        }
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_subblocks_k_dim);
    DeviceTimestampedData("Transaction size in bytes", k_subblock_size_bytes);
    // TX-only: sender iter = k_subblock_size_bytes; receiver iter = 16 (one atomic flit).
    constexpr uint32_t SEM_INC_BYTES = 16;
    uint32_t num_recv_iters = num_subblocks_k_dim - local_k_send_idx;
    uint32_t per_core_bytes = local_k_send_idx * k_subblock_size_bytes + num_recv_iters * SEM_INC_BYTES;
    DeviceTimestampedData("Per-core bytes", per_core_bytes);
}

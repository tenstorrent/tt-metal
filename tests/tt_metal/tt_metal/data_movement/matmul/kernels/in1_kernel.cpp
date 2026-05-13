// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "barrier_sync.hpp"

void kernel_main() {
    constexpr uint32_t test_id = get_compile_time_arg_val(0);
    constexpr uint32_t dram_bank_id = get_compile_time_arg_val(1);
    constexpr uint32_t physical_start_x = get_compile_time_arg_val(2);
    constexpr uint32_t physical_start_y = get_compile_time_arg_val(3);
    constexpr uint32_t physical_end_x = get_compile_time_arg_val(4);
    constexpr uint32_t physical_end_y = get_compile_time_arg_val(5);
    constexpr uint32_t num_cores_r_dim = get_compile_time_arg_val(6);
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t sender_valid_sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(9);

    uint32_t in1_per_core_dram_read_addr = get_arg_val<uint32_t>(0);   // sender: per-column DRAM offset
    uint32_t in1_per_core_read_size_bytes = get_arg_val<uint32_t>(1);  // sender: total bytes to read from DRAM
    uint32_t num_subblocks_k_dim = get_arg_val<uint32_t>(2);           // K iterations
    uint32_t k_subblock_size_bytes = get_arg_val<uint32_t>(3);         // bytes per K subblock
    uint32_t in1_l1_source_addr = get_arg_val<uint32_t>(4);            // sender: L1 buffer for DRAM read (full column)
    uint32_t in1_mcast_output_addr = get_arg_val<uint32_t>(5);         // all cores: multicast dest base
    // Barrier synchronization args
    uint32_t barrier_sem_id = get_arg_val<uint32_t>(6);
    uint32_t barrier_coord_x = get_arg_val<uint32_t>(7);
    uint32_t barrier_coord_y = get_arg_val<uint32_t>(8);
    uint32_t num_cores = get_arg_val<uint32_t>(9);
    uint32_t local_barrier_addr = get_arg_val<uint32_t>(10);
    uint32_t barrier_done_sem_id = get_arg_val<uint32_t>(11);

    uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    uint32_t sender_valid_sem_addr = get_semaphore(sender_valid_sem_id);
    uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);

    volatile tt_l1_ptr uint32_t* sender_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);

    bool is_sender = (my_y[0] == physical_start_y);

    // NOC1 has reversed routing, so swap start_y and end_y for the column multicast.
    uint64_t col_mcast_base = get_noc_multicast_addr(my_x[0], physical_end_y, my_x[0], physical_start_y, 0);
    uint64_t sender_sem_noc_addr = get_noc_addr(my_x[0], physical_start_y, sender_sem_addr);

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

    {
        DeviceZoneScopedN("RISCV1");

        // One big DRAM read pipelines better than K serialized small ones.
        if (is_sender) {
            uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, in1_per_core_dram_read_addr);
            noc_async_read(dram_noc_addr, in1_l1_source_addr, in1_per_core_read_size_bytes);
            noc_async_read_barrier();
        }

        for (uint32_t k = 0; k < num_subblocks_k_dim; k++) {
            uint32_t output_offset = k * k_subblock_size_bytes;
            uint32_t output_addr = in1_mcast_output_addr + output_offset;

            if (is_sender) {
                uint32_t src_addr = in1_l1_source_addr + output_offset;

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
                noc_semaphore_inc(sender_sem_noc_addr, 1);
            }

            noc_semaphore_wait(receiver_sem_ptr, 1);
            noc_semaphore_set(receiver_sem_ptr, 0);
        }
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_subblocks_k_dim);
    DeviceTimestampedData("Transaction size in bytes", k_subblock_size_bytes);
    // TX-only: sender K * k_subblock_size_bytes; receiver K * 16 (one atomic flit each).
    constexpr uint32_t SEM_INC_BYTES = 16;
    uint32_t per_core_bytes =
        is_sender ? (num_subblocks_k_dim * k_subblock_size_bytes) : (num_subblocks_k_dim * SEM_INC_BYTES);
    DeviceTimestampedData("Per-core bytes", per_core_bytes);
}

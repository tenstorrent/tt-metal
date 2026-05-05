// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "barrier_sync.hpp"

// in1 path for 1D / 1D v2 matmul.
//
// Row 0 is the fixed sender for every column. Once at the start of the timer zone
// it reads the ENTIRE in1 column slice from DRAM into a local L1 source buffer
// (one big noc_async_read of in1_per_core_read_size_bytes), then iterates K times
// multicasting each k_subblock_size_bytes chunk down its column. Rows 1..R-1 only
// signal readiness and wait — they never touch DRAM.
//
// One big DRAM read amortizes setup overhead and pipelines better than K small
// reads. After K multicasts every core in the column has the full in1 column
// data assembled in its multicast output buffer.
//
// This eliminates the per-core DRAM bottleneck of the previous "every core reads
// DRAM" approach: only C cores (one per column, on row 0) hit DRAM, and the
// remaining R-1 cores per column receive the data via L1-to-L1 NOC multicast.
//
// Single-row case (R=1): every core is its own row-0 sender. Each core reads DRAM
// independently; no multicast is needed. Falls back to a unicast self-write into
// the mcast output buffer (HW limitation with single-core multicast loopback).
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

    // Row 0 is the fixed sender (reads DRAM once, then multicasts K subblocks down its column).
    bool is_sender = (my_y[0] == physical_start_y);

    // Column-wise multicast addresses. NOC1 (RISCV_1) has reversed routing direction,
    // so swap start_y and end_y when constructing the multicast destination.
    uint64_t col_mcast_base = get_noc_multicast_addr(my_x[0], physical_end_y, my_x[0], physical_start_y, 0);

    // Address of row-0 sender's sender_sem in this column (used by receivers to signal readiness).
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

        // Phase 1 (sender only): read the entire in1 column slice from DRAM in ONE call.
        // Lets DRAM pipeline a single large transfer instead of K serialized small ones.
        if (is_sender) {
            uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, in1_per_core_dram_read_addr);
            noc_async_read(dram_noc_addr, in1_l1_source_addr, in1_per_core_read_size_bytes);
            noc_async_read_barrier();
        }

        // Phase 2: K-loop multicasts each subblock from the source buffer down the column.
        for (uint32_t k = 0; k < num_subblocks_k_dim; k++) {
            uint32_t output_offset = k * k_subblock_size_bytes;
            uint32_t output_addr = in1_mcast_output_addr + output_offset;

            if (is_sender) {
                // Source partition for this K iteration: contiguous within the pre-read buffer.
                uint32_t src_addr = in1_l1_source_addr + output_offset;

                // Wait for all R-1 receivers in the column to signal readiness.
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
                    // Single row (R=1): unicast self-write (HW limitation with single-core multicast loopback).
                    uint64_t local_dest_addr = get_noc_addr(my_x[0], my_y[0], output_addr);
                    noc_async_write(src_addr, local_dest_addr, k_subblock_size_bytes);
                    noc_async_write_barrier();
                    noc_semaphore_set(receiver_sem_ptr, 1);
                }
            } else {
                // RECEIVER: signal readiness to the row-0 sender in this column.
                noc_semaphore_inc(sender_sem_noc_addr, 1);
            }

            // All cores wait for data arrival, then reset for next iteration.
            noc_semaphore_wait(receiver_sem_ptr, 1);
            noc_semaphore_set(receiver_sem_ptr, 0);
        }
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_subblocks_k_dim);
    DeviceTimestampedData("Transaction size in bytes", k_subblock_size_bytes);
    // Per-core actual NOC bytes pushed by this core's RISCV_1 across the K-loop.
    // Row-0 sender multicasts k_subblock_size_bytes K times; receivers issue K
    // semaphore incs (one atomic op flit each). DRAM read RX is not counted in the
    // TX-only convention shared with the other matmul kernels.
    constexpr uint32_t SEM_INC_BYTES = 16;
    uint32_t per_core_bytes =
        is_sender ? (num_subblocks_k_dim * k_subblock_size_bytes) : (num_subblocks_k_dim * SEM_INC_BYTES);
    DeviceTimestampedData("Per-core bytes", per_core_bytes);
}

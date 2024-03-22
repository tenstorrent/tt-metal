// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    constexpr uint32_t num_transfers = get_compile_time_arg_val(0);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_pages = get_compile_time_arg_val(3);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(4);
    constexpr uint32_t eth_receiver_noc_x = get_compile_time_arg_val(5);
    constexpr uint32_t eth_receiver_noc_y = get_compile_time_arg_val(6);
    constexpr uint32_t eth_receiver_l1_semaphore_addr = get_compile_time_arg_val(7);
    constexpr uint32_t receiver_read_sem_addr = get_compile_time_arg_val(8);
    constexpr uint32_t half_cb_n_pages = get_compile_time_arg_val(9);
    static_assert (half_cb_n_pages > rem_num_pages, "half_cb_n_pages must be greater than 0");

    const uint32_t eth_receiver_l1_base_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    // Eth receiver will set this semaphore when data is available
    volatile tt_l1_ptr uint32_t* receiver_read_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_read_sem_addr);

    uint32_t total_num_transfers = (num_full_chunks + (rem_num_pages > 0 ? 1 : 0)) * num_transfers;
    uint32_t eth_core = eth_receiver_noc_x | (eth_receiver_noc_y << 16);
    uint32_t transfers_completed = 0;
    // Address of the buffer on the eth receiver, this is different per receiver worker core
    const uint64_t eth_receiver_l1_base_noc_addr = get_noc_addr(eth_receiver_noc_x, eth_receiver_noc_y, eth_receiver_l1_base_addr);
    // Address of the semaphore on the eth receiver, this is the same per receiver worker core
    const uint64_t eth_receiver_l1_semaphore_noc_addr = get_noc_addr(eth_receiver_noc_x, eth_receiver_noc_y, eth_receiver_l1_semaphore_addr);
    for (uint32_t i = 0; i < num_transfers; ++i) {
        if constexpr (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                uint64_t eth_receiver_l1_curr_noc_addr = eth_receiver_l1_base_noc_addr;
                noc_semaphore_wait(receiver_read_semaphore_addr_ptr, 1);
                noc_semaphore_set(receiver_read_semaphore_addr_ptr, 0);
                // Read page by page so that writer can be kicked off instead of being blocked waiting for full chunk to be read
                // Look into perf/optimizations for this
                fetch_chunk(cb_id_in0, num_pages, page_size, eth_receiver_l1_base_noc_addr);
                noc_semaphore_inc(eth_receiver_l1_semaphore_noc_addr, 1);
                transfers_completed++;
            }
        }
        if constexpr (rem_num_pages > 0) {
            uint64_t eth_receiver_l1_curr_noc_addr = eth_receiver_l1_base_noc_addr;
            noc_semaphore_wait(receiver_read_semaphore_addr_ptr, 1);
            noc_semaphore_set(receiver_read_semaphore_addr_ptr, 0);
            fetch_chunk(cb_id_in0, rem_num_pages, page_size, eth_receiver_l1_base_noc_addr);
            noc_semaphore_inc(eth_receiver_l1_semaphore_noc_addr, 1);
            ASSERT(num_pages == 0 || num_pages > rem_num_pages);
            ASSERT(half_cb_n_pages > rem_num_pages);
            push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
            transfers_completed++;
        }
    }

}

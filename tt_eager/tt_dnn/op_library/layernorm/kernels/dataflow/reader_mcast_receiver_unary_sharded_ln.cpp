// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

// split REDUCE across cores
void kernel_main() {

    constexpr uint32_t reduce_receiver_semaphore_addr  = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_addr    = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks                      = get_compile_time_arg_val(2);
    constexpr uint32_t block_h                         = get_compile_time_arg_val(3);
    const bool is_all_to_all_worker                    = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t num_all_to_all_workers          = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_per_worker            = get_compile_time_arg_val(6);
    constexpr uint32_t num_tiles_per_worker_last       = get_compile_time_arg_val(7);
    constexpr bool row_major                           = (bool) get_compile_time_arg_val(8);
    constexpr uint32_t num_x                           = get_compile_time_arg_val(9);
    constexpr uint32_t num_y                           = get_compile_time_arg_val(10);

    const bool is_last_all_to_all_worker                = get_arg_val<uint32_t>(0);
    const uint32_t all_to_all_tile_offset_bytes         = get_arg_val<uint32_t>(1);
    const uint32_t start_x                              = get_arg_val<uint32_t>(2);
    const uint32_t start_y                              = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t * in0_remote_noc_x          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(4));
    volatile tt_l1_ptr uint32_t * in0_remote_noc_y          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(4 + num_x));

    const uint32_t num_tiles_to_read = is_last_all_to_all_worker ? num_tiles_per_worker_last : num_tiles_per_worker;

    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0; // E[x] partial reduce
    constexpr uint32_t cb_ex = tt::CB::dataflow1; // E[x] global reduce
    constexpr uint32_t cb_ex_external = tt::CB::dataflow2;
    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3; // E[(x-E[x])^2] partial reduce
    constexpr uint32_t cb_ex2 = tt::CB::dataflow4; // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2 = tt::CB::dataflow5;
    constexpr uint32_t cb_ex2pe = tt::CB::c_intermed3;
    constexpr uint32_t cb_ex_global = tt::CB::dataflow7; // E[x] global reduce

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial2); // tile size
    const DataFormat data_format = get_dataformat(cb_ex_partial2); // data format

    uint64_t remote_noc_addrs[is_all_to_all_worker ? num_blocks : 1];
    if constexpr (is_all_to_all_worker) {
        uint32_t x = start_x, y = start_y;
        for (uint32_t i = 0; i < num_blocks; ++i) {
            remote_noc_addrs[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
            if constexpr(row_major) {
                ++x;
                if (x == num_x) {
                    x = 0;
                    ++y;
                    if (y == num_y) {
                        y = 0;
                    }
                }
            } else {
                ++y;
                if (y == num_y) {
                    y = 0;
                    ++x;
                    if (x == num_x) {
                        x = 0;
                    }
                }
            }
        }
    } else {
        remote_noc_addrs[0] = get_noc_addr(in0_remote_noc_x[0], in0_remote_noc_y[0], 0);
    }

    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);

    const uint64_t reduce_receiver_semaphore_noc_addr = get_noc_addr(in0_remote_noc_x[0], in0_remote_noc_y[0], reduce_receiver_semaphore_addr);

    const auto& global_reduce_receiver = [&](const uint32_t cb_partial, const uint32_t cb_external, const uint32_t cb_ex, const uint32_t cb_ex_global) __attribute__((always_inline))
    {
        // global reduce
        // wait for local data ready
        cb_wait_front(cb_partial, block_h);

        // inc top core
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);

        if constexpr (is_all_to_all_worker) {
            // read data from other cores
            uint32_t l1_read_addr_ex_par = get_read_ptr(cb_partial);
            l1_read_addr_ex_par += all_to_all_tile_offset_bytes;
            for (uint32_t i = 0; i < num_tiles_to_read; i++) {
                uint32_t l1_write_addr_external = get_write_ptr(cb_external);
                for(uint32_t block = 0; block < num_blocks; block++) {
                    cb_reserve_back(cb_external, 1);
                    uint64_t noc_addr_ex_par = remote_noc_addrs[block] | l1_read_addr_ex_par;
                    noc_async_read_one_packet(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
                    l1_write_addr_external += single_tile_size_bytes;

                    noc_async_read_barrier();
                    cb_push_back(cb_external, 1);
                }
                l1_read_addr_ex_par += single_tile_size_bytes;
            }

            // send result to other cores
            cb_wait_front(cb_ex, num_tiles_to_read);
        }

        // sync with other workers
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);

        for (uint32_t block = 0; block < num_all_to_all_workers; ++block) {
            uint32_t num_tiles = block == num_all_to_all_workers - 1 ? num_tiles_per_worker_last : num_tiles_per_worker;
            cb_reserve_back(cb_ex_global, num_tiles);
            noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, block+1);
            cb_push_back(cb_ex_global, num_tiles);
        }
    };
    #ifndef RMSNORM
    global_reduce_receiver(cb_ex_partial, cb_ex_external, cb_ex, cb_ex_global);
    #endif
    global_reduce_receiver(cb_ex_partial2, cb_ex_external2, cb_ex2pe, cb_ex_global);

}

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug/dprint.h"

// split REDUCE across cores
void kernel_main() {

    constexpr uint32_t reduce_receiver_semaphore_addr  = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_addr    = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks                      = get_compile_time_arg_val(2);
    constexpr uint32_t block_h                         = get_compile_time_arg_val(3);
    const bool is_all_to_all_worker                    = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t num_all_to_all_workers          = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_per_worker            = get_compile_time_arg_val(6);

    const uint32_t all_to_all_tile_offset_bytes         = get_arg_val<uint32_t>(0);
    const uint32_t noc_same_coord                       = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t * noc_diff_coord        = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(2));

    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0; // E[x] partial reduce
    constexpr uint32_t cb_ex = tt::CB::dataflow1; // E[x] global reduce
    constexpr uint32_t cb_ex_external = tt::CB::dataflow2;
    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3; // E[(x-E[x])^2] partial reduce
    constexpr uint32_t cb_ex2 = tt::CB::dataflow4; // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2 = tt::CB::dataflow5;
    constexpr uint32_t cb_ex2pe = tt::CB::c_intermed3;
    constexpr uint32_t cb_ex_global = tt::CB::dataflow7; // E[x] global reduce

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial); // tile size
    const DataFormat data_format = get_dataformat(cb_ex_partial); // data format

    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);

    // global reduce
    if constexpr(is_all_to_all_worker) {
        // wait for local data ready
        cb_wait_front(cb_ex_partial, block_h);

        // inc top core
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        const uint64_t reduce_receiver_semaphore_noc_addr = get_noc_addr(noc_same_coord, 1, reduce_receiver_semaphore_addr);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);

        // read data from other cores
        uint32_t l1_read_addr_ex_par = get_read_ptr(cb_ex_partial);
        l1_read_addr_ex_par += all_to_all_tile_offset_bytes;
        for (uint32_t i = 0; i < num_tiles_per_worker; i++) {
            cb_reserve_back(cb_ex_external, num_blocks);
            uint32_t l1_write_addr_external = get_write_ptr(cb_ex_external);
            for(uint32_t block = 0; block < num_blocks; block++) {
                uint64_t noc_addr_ex_par = get_noc_addr(noc_same_coord, noc_diff_coord[block], l1_read_addr_ex_par);
                noc_async_read(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
                l1_write_addr_external += single_tile_size_bytes;
            }
            l1_read_addr_ex_par += single_tile_size_bytes;
            noc_async_read_barrier();
            cb_push_back(cb_ex_external, num_blocks);
        }

        // send result to other cores
        uint32_t l1_write_addr_ex = get_write_ptr(cb_ex);
        uint32_t l1_write_addr_ex_global = get_write_ptr(cb_ex_global);
        cb_wait_front(cb_ex, num_tiles_per_worker);

        // sync with other workers
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);

        for (uint32_t block = 0; block < num_all_to_all_workers; block++) {
            cb_reserve_back(cb_ex_global, num_tiles_per_worker);
            noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, block+1);
            cb_push_back(cb_ex_global, num_tiles_per_worker);
        }

    } else {
        // wait for local data ready
        cb_wait_front(cb_ex_partial, block_h);

        // inc top core
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        const uint64_t reduce_receiver_semaphore_noc_addr = get_noc_addr(noc_same_coord, 1, reduce_receiver_semaphore_addr);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);

        // sync with other workers
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);

        for (uint32_t block = 0; block < num_all_to_all_workers; block++) {
            cb_reserve_back(cb_ex_global, num_tiles_per_worker);
            noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, block+1);
            cb_push_back(cb_ex_global, num_tiles_per_worker);
        }

    };

    // global reduce
    if constexpr(is_all_to_all_worker) {
        // wait for local data ready
        cb_wait_front(cb_ex_partial2, block_h);
        // inc semaphore of other cores
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        const uint64_t reduce_receiver_semaphore_noc_addr = get_noc_addr(noc_same_coord, 1, reduce_receiver_semaphore_addr);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);

        // read data from other cores
        uint32_t l1_read_addr_ex_par = get_read_ptr(cb_ex_partial2);
        l1_read_addr_ex_par += all_to_all_tile_offset_bytes;
        for (uint32_t i = 0; i < num_tiles_per_worker; i++) {
            cb_reserve_back(cb_ex_external2, num_blocks);
            uint32_t l1_write_addr_external = get_write_ptr(cb_ex_external2);
            for(uint32_t block = 0; block < num_blocks; block++) {
                uint64_t noc_addr_ex_par = get_noc_addr(noc_same_coord, noc_diff_coord[block], l1_read_addr_ex_par);
                noc_async_read(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
                l1_write_addr_external += single_tile_size_bytes;
            }
            l1_read_addr_ex_par += single_tile_size_bytes;
            noc_async_read_barrier();
            cb_push_back(cb_ex_external2, num_blocks);
        }

        // send result to other cores
        uint32_t l1_write_addr_ex = get_write_ptr(cb_ex2pe);
        uint32_t l1_write_addr_ex_global = get_write_ptr(cb_ex_global);
        cb_wait_front(cb_ex2pe, num_tiles_per_worker);

        // sync with other workers
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);

        for (uint32_t block = 0; block < num_all_to_all_workers; block++) {
            cb_reserve_back(cb_ex_global, num_tiles_per_worker);
            noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, block+1);
            cb_push_back(cb_ex_global, num_tiles_per_worker);
        }

    } else {
        // wait for local data ready
        cb_wait_front(cb_ex_partial2, block_h);
        // inc semaphore of other cores
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        const uint64_t reduce_receiver_semaphore_noc_addr = get_noc_addr(noc_same_coord, 1, reduce_receiver_semaphore_addr);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);

        // // sync with other workers
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);

        for (uint32_t block = 0; block < num_all_to_all_workers; block++) {
            cb_reserve_back(cb_ex_global, num_tiles_per_worker);
            noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, block+1);
            cb_push_back(cb_ex_global, num_tiles_per_worker);
        }

    };


}

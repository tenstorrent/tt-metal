// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

// #include "debug/dprint.h"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t reduce_receiver_semaphore_addr  = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_addr    = get_compile_time_arg_val(1);
    constexpr uint32_t num_group_batch                 = get_compile_time_arg_val(2);

    const uint32_t mcast_sender_noc_x         = get_arg_val<uint32_t>(0);
    const uint32_t mcast_sender_noc_y         = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0; // E[x] partial reduce
    constexpr uint32_t cb_ex_global = tt::CB::dataflow7; // E[x] global reduce

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial); // tile size
    const DataFormat data_format = get_dataformat(cb_ex_partial); // data format

    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);

    const uint64_t reduce_receiver_semaphore_noc_addr = get_noc_addr(mcast_sender_noc_x, mcast_sender_noc_y, reduce_receiver_semaphore_addr);

    for (uint32_t i=0; i < 2; ++i) {
        // wait for local data ready
        cb_wait_front(cb_ex_partial, num_group_batch);
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        cb_reserve_back(cb_ex_global, num_group_batch);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
        cb_push_back(cb_ex_global, num_group_batch);
        cb_pop_front(cb_ex_partial, num_group_batch);
    }
}

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
// #include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {

    uint32_t receiver_noc_x          = get_arg_val<uint32_t>(0);
    uint32_t receiver_noc_y          = get_arg_val<uint32_t>(1);
    uint32_t num_tiles               = get_arg_val<uint32_t>(2);
    uint32_t sender_semaphore_addr   = get_semaphore(get_arg_val<uint32_t>(3));
    uint32_t receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(4));
    uint32_t l1_valid_value_addr     = get_semaphore(get_arg_val<uint32_t>(5));
    uint32_t num_repetitions         = get_arg_val<uint32_t>(6);

    // initialized by the host to 0 before program launch
    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    // local valid value in L1
    volatile tt_l1_ptr uint32_t* l1_valid_value_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_valid_value_addr);
    *(l1_valid_value_addr_ptr) = VALID;

    constexpr uint32_t cb_id            = get_compile_time_arg_val(0);
    constexpr uint32_t block_size_tiles = get_compile_time_arg_val(1);

    uint32_t block_size_bytes = get_tile_size(cb_id) * block_size_tiles;

    uint64_t receiver_semaphore_noc_addr = get_noc_addr(receiver_noc_x, receiver_noc_y, receiver_semaphore_addr);

    for (uint32_t j = 0; j < num_repetitions; j++) {
        for (uint32_t i = 0; i < num_tiles; i += block_size_tiles) {

            // wait until receiver has set the sender's semaphore_addr value to 1, which means receiver has reserved space in the CB
            noc_semaphore_wait(sender_semaphore_addr_ptr, 1);

            if (i > 0) {
                cb_pop_front(cb_id, block_size_tiles);
            }
            cb_wait_front(cb_id, block_size_tiles);
            uint32_t l1_addr = get_read_ptr(cb_id);

            // now we have the block in the CB (at l1_addr), we can send to receiver
            uint64_t receiver_data_noc_addr      = get_noc_addr(receiver_noc_x, receiver_noc_y, l1_addr);
            noc_async_write(l1_addr, receiver_data_noc_addr, block_size_bytes);

            // set the sender's semaphore value back to zero for the next block
            // we need to reset before we set the receiver's semaphore
            noc_semaphore_set(sender_semaphore_addr_ptr, 0);

            // we now set the receiver's semaphore, so that it knows that the data has been written to the CB
            // must use noc_semaphore_set_remote and not noc_semaphore_inc in the sender
            // because we need to ensure that data is written to the remote CB before we set the semaphore
            // noc_async_write and noc_semaphore_set_remote are ordered
            noc_semaphore_set_remote(l1_valid_value_addr, receiver_semaphore_noc_addr);

            // this barrier is not needed, sempahore inter-lock already guarantees that we won't overwrite local CB with new data
            // ie, it is safe to pop here, because the data in the CB won't actually be overwritten until the receiver has set the semaphore (which means it was received)
            // this barrier would hurt performance for smaller transfers (<16KB), but for larger transfers it wouldn't make a difference
            // noc_async_write_barrier();
        }
        cb_pop_front(cb_id, block_size_tiles);
    }
}

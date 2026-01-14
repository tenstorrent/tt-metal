// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"

void kernel_main() {
    constexpr uint32_t mcast_cb = get_compile_time_arg_val(0);
    constexpr uint32_t mcast_sender_cb = get_compile_time_arg_val(1);
    constexpr uint32_t num_senders = get_compile_time_arg_val(2);

    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(3));
    volatile tt_l1_ptr uint32_t* mcast_receiver_semaphore_addr_ptr =
        (volatile tt_l1_ptr uint32_t*)mcast_receiver_semaphore_addr;

    const uint32_t mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(4));
    volatile tt_l1_ptr uint32_t* mcast_sender_semaphore_addr_ptr =
        (volatile tt_l1_ptr uint32_t*)mcast_sender_semaphore_addr;
    constexpr uint64_t mcast_sender_noc_coord_x_start = get_compile_time_arg_val(5);
    constexpr uint64_t mcast_sender_noc_coord_y_start = get_compile_time_arg_val(6);
    constexpr uint64_t mcast_sender_noc_coord_x_end = get_compile_time_arg_val(7);
    constexpr uint64_t mcast_sender_noc_coord_y_end = get_compile_time_arg_val(8);

    // Wait for all senders to finish sending data
    noc_semaphore_wait(mcast_receiver_semaphore_addr_ptr, num_senders);
    noc_async_posted_writes_flushed();

    // Mcast to all cores and then update semaphore
    const uint64_t mcast_sender_noc_addr = get_noc_multicast_addr(
        mcast_sender_noc_coord_x_start,
        mcast_sender_noc_coord_y_start,
        mcast_sender_noc_coord_x_end,
        mcast_sender_noc_coord_y_end,
        get_write_ptr(mcast_sender_cb));
    noc_async_write_multicast(
        get_read_ptr(mcast_cb), mcast_sender_noc_addr, get_tile_size(mcast_sender_cb) * num_senders, num_senders);
    const uint64_t mcast_sender_semaphore_noc_addr = get_noc_multicast_addr(
        mcast_sender_noc_coord_x_start,
        mcast_sender_noc_coord_y_start,
        mcast_sender_noc_coord_x_end,
        mcast_sender_noc_coord_y_end,
        mcast_sender_semaphore_addr);
    noc_semaphore_inc(mcast_sender_semaphore_noc_addr, VALID);
    noc_async_posted_writes_flushed();

    DPRINT << "Semaphore set" << ENDL();

    DPRINT << "Mcast reader kernel finished" << ENDL();
}

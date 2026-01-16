// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include <tools/profiler/kernel_profiler.hpp>

#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"

FORCE_INLINE void wait_for_gather(
    volatile tt_l1_ptr uint32_t* mcast_receiver_semaphore_addr_ptr, uint32_t num_senders, uint32_t debug_enabled) {
    DeviceZoneScopedN("wait_for_gather");
    // Wait for all senders to finish sending data
    noc_semaphore_wait(mcast_receiver_semaphore_addr_ptr, num_senders);

    // Reset the local semaphore for reuse
    noc_semaphore_set(mcast_receiver_semaphore_addr_ptr, 0);
}

FORCE_INLINE void mcast(
    uint32_t mcast_cb,
    uint32_t mcast_sender_cb,
    uint32_t num_senders,
    uint32_t mcast_sender_semaphore_addr,
    uint64_t mcast_sender_noc_coord_x_start,
    uint64_t mcast_sender_noc_coord_y_start,
    uint64_t mcast_sender_noc_coord_x_end,
    uint64_t mcast_sender_noc_coord_y_end,
    uint32_t debug_enabled) {
    DeviceZoneScopedN("mcast");
    // Mcast to all cores and then update semaphore
    const uint64_t mcast_sender_noc_addr = get_noc_multicast_addr(
        mcast_sender_noc_coord_x_start,
        mcast_sender_noc_coord_y_start,
        mcast_sender_noc_coord_x_end,
        mcast_sender_noc_coord_y_end,
        get_write_ptr(mcast_sender_cb));
    noc_async_write_multicast(
        get_read_ptr(mcast_cb), mcast_sender_noc_addr, get_tile_size(mcast_sender_cb) * num_senders, num_senders);

    // Set up local L1 scratch to hold VALID value for multicast semaphore set
    uint32_t semaphore_valid_addr = get_write_ptr(mcast_sender_cb);
    volatile tt_l1_ptr uint32_t* semaphore_valid_addr_ptr = (volatile tt_l1_ptr uint32_t*)semaphore_valid_addr;
    semaphore_valid_addr_ptr[0] = VALID;

    // Multicast semaphore set to all sender cores
    const uint64_t mcast_sender_semaphore_noc_addr = get_noc_multicast_addr(
        mcast_sender_noc_coord_x_start,
        mcast_sender_noc_coord_y_start,
        mcast_sender_noc_coord_x_end,
        mcast_sender_noc_coord_y_end,
        mcast_sender_semaphore_addr);
    noc_semaphore_set_multicast(semaphore_valid_addr, mcast_sender_semaphore_noc_addr, num_senders);
    noc_async_posted_writes_flushed();
}

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
    constexpr uint32_t debug_enabled = get_compile_time_arg_val(9);
    constexpr uint32_t num_layers = get_compile_time_arg_val(10);

    for (uint32_t layer = 0; layer < num_layers; layer++) {
        // Between matmul+relu and matmul+bias we gather+mcast the result
        wait_for_gather(mcast_receiver_semaphore_addr_ptr, num_senders, debug_enabled);
        mcast(
            mcast_cb,
            mcast_sender_cb,
            num_senders,
            mcast_sender_semaphore_addr,
            mcast_sender_noc_coord_x_start,
            mcast_sender_noc_coord_y_start,
            mcast_sender_noc_coord_x_end,
            mcast_sender_noc_coord_y_end,
            debug_enabled);

        // After matmul+bias we gather+mcast the result
        wait_for_gather(mcast_receiver_semaphore_addr_ptr, num_senders, debug_enabled);
        mcast(
            mcast_cb,
            mcast_sender_cb,
            num_senders,
            mcast_sender_semaphore_addr,
            mcast_sender_noc_coord_x_start,
            mcast_sender_noc_coord_y_start,
            mcast_sender_noc_coord_x_end,
            mcast_sender_noc_coord_y_end,
            debug_enabled);
    }
}

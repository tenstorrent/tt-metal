// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include <tools/profiler/kernel_profiler.hpp>

#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"

FORCE_INLINE void wait_for_gather(
    volatile tt_l1_ptr uint32_t* mcast_receiver_semaphore_addr_ptr, uint32_t num_senders) {
    DeviceZoneScopedN("wait_for_gather");

    // Wait for all senders to finish sending data
    noc_semaphore_wait(mcast_receiver_semaphore_addr_ptr, num_senders);

    // Reset the local semaphore for reuse
    noc_semaphore_set(mcast_receiver_semaphore_addr_ptr, 0);
}

FORCE_INLINE void mcast(
    uint32_t mcast_cb,
    uint32_t mcast_dest_base_addr,
    uint32_t num_senders,
    uint32_t mcast_sender_semaphore_addr,
    uint64_t mcast_sender_noc_coord_x_start,
    uint64_t mcast_sender_noc_coord_y_start,
    uint64_t mcast_sender_noc_coord_x_end,
    uint64_t mcast_sender_noc_coord_y_end) {
    DeviceZoneScopedN("mcast");
    // Mcast to all cores and then update semaphore
    const uint64_t mcast_sender_noc_addr = get_noc_multicast_addr(
        mcast_sender_noc_coord_x_start,
        mcast_sender_noc_coord_y_start,
        mcast_sender_noc_coord_x_end,
        mcast_sender_noc_coord_y_end,
        mcast_dest_base_addr);
    // Use mcast_cb tile size since source and destination have the same tile size
    noc_async_write_multicast(
        get_read_ptr(mcast_cb),
        mcast_sender_noc_addr,
        get_tile_size(mcast_cb) * num_senders,
        num_senders,
        false /* linked */);

    // Use L1 scratch to hold VALID value for multicast semaphore set
    uint32_t semaphore_valid_addr = mcast_dest_base_addr;
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
    constexpr uint32_t mm2_full_cb = get_compile_time_arg_val(1);
    constexpr uint32_t mm1_full_cb = get_compile_time_arg_val(2);
    constexpr uint32_t input_buffer_base_addr = get_compile_time_arg_val(3);
    constexpr uint32_t num_senders = get_compile_time_arg_val(4);

    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(5));
    volatile tt_l1_ptr uint32_t* mcast_receiver_semaphore_addr_ptr =
        (volatile tt_l1_ptr uint32_t*)mcast_receiver_semaphore_addr;

    const uint32_t mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(6));
    volatile tt_l1_ptr uint32_t* mcast_sender_semaphore_addr_ptr =
        (volatile tt_l1_ptr uint32_t*)mcast_sender_semaphore_addr;
    constexpr uint64_t mcast_sender_noc_coord_x_start = get_compile_time_arg_val(7);
    constexpr uint64_t mcast_sender_noc_coord_y_start = get_compile_time_arg_val(8);
    constexpr uint64_t mcast_sender_noc_coord_x_end = get_compile_time_arg_val(9);
    constexpr uint64_t mcast_sender_noc_coord_y_end = get_compile_time_arg_val(10);
    constexpr uint32_t num_layers = get_compile_time_arg_val(11);

    for (uint32_t layer = 0; layer < num_layers; layer++) {
        DeviceZoneScopedN("gather_and_mcast");

        // First mcast: matmul+relu result -> MM2_FULL_CB
        // Use get_write_ptr for mm2_full_cb since it's not bound to input tensor
        wait_for_gather(mcast_receiver_semaphore_addr_ptr, num_senders);
        {
            const uint32_t mm2_base_addr = get_write_ptr(mm2_full_cb);
            mcast(
                mcast_cb,
                mm2_base_addr,
                num_senders,
                mcast_sender_semaphore_addr,
                mcast_sender_noc_coord_x_start,
                mcast_sender_noc_coord_y_start,
                mcast_sender_noc_coord_x_end,
                mcast_sender_noc_coord_y_end);
        }

        // Second mcast: matmul+bias result -> MM1_FULL_CB
        // Use input buffer base address for mm1_full_cb since it's bound to input tensor
        wait_for_gather(mcast_receiver_semaphore_addr_ptr, num_senders);
        mcast(
            mcast_cb,
            input_buffer_base_addr,
            num_senders,
            mcast_sender_semaphore_addr,
            mcast_sender_noc_coord_x_start,
            mcast_sender_noc_coord_y_start,
            mcast_sender_noc_coord_x_end,
            mcast_sender_noc_coord_y_end);
    }
}

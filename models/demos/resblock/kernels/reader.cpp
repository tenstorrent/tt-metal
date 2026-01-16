// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"

template <uint32_t CbOut, uint32_t NumTiles>
FORCE_INLINE void wait_for_mcast(volatile tt_l1_ptr uint32_t* mcast_sender_semaphore_addr_ptr, uint32_t debug_enabled) {
    if (debug_enabled) {
        DPRINT << "mcast: wait sender (" << (uint32_t)get_absolute_logical_x() << ","
               << (uint32_t)get_absolute_logical_y() << ")" << ENDL();
    }
    noc_semaphore_wait(mcast_sender_semaphore_addr_ptr, VALID);
    cb_push_back(CbOut, NumTiles);
    noc_semaphore_set(mcast_sender_semaphore_addr_ptr, 0);
    if (debug_enabled) {
        DPRINT << "mcast: got sender (" << (uint32_t)get_absolute_logical_x() << ","
               << (uint32_t)get_absolute_logical_y() << ")" << ENDL();
    }
}

template <
    uint32_t CbIn,
    uint32_t CbOut,
    uint32_t NumTiles,
    uint32_t MCastReceiverNocX,
    uint32_t MCastReceiverNocY,
    uint32_t MCastReceiverSemaphoreId>
void gather(uint32_t receiver_data_addr, uint32_t offset) {
    cb_reserve_back(CbOut, NumTiles);
    cb_wait_front(CbIn, NumTiles);

    // Write all tiles from CbIn to CbOut using noc_async_write (will remove once the gather+mcast is implemented)
    // noc_async_write(get_read_ptr(CbIn), get_noc_addr(get_write_ptr(CbOut)), get_tile_size(CbIn) * NumTiles);

    // Gather to receiver core (write and then signal using semaphore)
    const uint64_t mcast_reciever_noc_coord = get_noc_addr(MCastReceiverNocX, MCastReceiverNocY, 0);
    const uint64_t mcast_reciever_noc_addr = mcast_reciever_noc_coord | (uint64_t)(receiver_data_addr + offset);
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(MCastReceiverSemaphoreId);
    const uint64_t mcast_receiver_semaphore_noc_addr =
        mcast_reciever_noc_coord | (uint64_t)mcast_receiver_semaphore_addr;

    noc_async_write_one_packet<true, true>(get_read_ptr(CbIn), mcast_reciever_noc_addr, get_tile_size(CbIn) * NumTiles);
    noc_semaphore_inc<true>(mcast_receiver_semaphore_noc_addr, 1);
    noc_async_posted_writes_flushed();

    // cb_push_back(CbOut, NumTiles); Don't pop because we want to wait for mcast to finish before starting second
    // matmul
    cb_pop_front(CbIn, NumTiles);
}

template <uint32_t SenderLogicalXStart, uint32_t SenderLogicalYStart, uint32_t SenderGridWidth>
FORCE_INLINE uint32_t compute_sender_tile_offset_bytes(uint32_t tile_size_bytes) {
    const uint32_t sender_logical_x = get_absolute_logical_x();
    const uint32_t sender_logical_y = get_absolute_logical_y();
    const uint32_t sender_tile_index =
        (sender_logical_y - SenderLogicalYStart) * SenderGridWidth + (sender_logical_x - SenderLogicalXStart);
    return sender_tile_index * tile_size_bytes;
}

void kernel_main() {
    constexpr uint32_t in0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t weight0_cb = get_compile_time_arg_val(1);
    constexpr uint32_t weight1_cb = get_compile_time_arg_val(2);
    constexpr uint32_t intermediate_pregather_cb = get_compile_time_arg_val(3);
    constexpr uint32_t intermediate_full_cb = get_compile_time_arg_val(4);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(5);

    constexpr uint32_t mcast_receiver_noc_x = get_compile_time_arg_val(6);
    constexpr uint32_t mcast_receiver_noc_y = get_compile_time_arg_val(7);
    constexpr uint32_t mcast_receiver_semaphore_id = get_compile_time_arg_val(8);
    constexpr uint32_t mcast_reciever_cb = get_compile_time_arg_val(9);

    const uint32_t mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(10));
    volatile tt_l1_ptr uint32_t* mcast_sender_semaphore_addr_ptr =
        (volatile tt_l1_ptr uint32_t*)mcast_sender_semaphore_addr;

    constexpr uint32_t sender_logical_x_start = get_compile_time_arg_val(11);
    constexpr uint32_t sender_logical_y_start = get_compile_time_arg_val(12);
    constexpr uint32_t sender_grid_width = get_compile_time_arg_val(13);
    constexpr uint32_t debug_enabled = get_compile_time_arg_val(14);

    const uint32_t mcast_reciever_base_address = get_write_ptr(mcast_reciever_cb);

    constexpr uint32_t num_output_tiles = 1;  // Set to 1 because we only iterate over K tiles at a time

    cb_reserve_back(in0_cb, num_tiles_k);
    cb_push_back(in0_cb, num_tiles_k);

    cb_reserve_back(weight0_cb, num_tiles_k);
    cb_push_back(weight0_cb, num_tiles_k);

    cb_reserve_back(weight1_cb, num_tiles_k);
    cb_push_back(weight1_cb, num_tiles_k);

    // Gather after first matmul so that we can mcast full result to all cores
    const uint32_t gather_destination_tile_offset_bytes =
        compute_sender_tile_offset_bytes<sender_logical_x_start, sender_logical_y_start, sender_grid_width>(
            get_tile_size(intermediate_pregather_cb));
    gather<
        intermediate_pregather_cb,
        intermediate_full_cb,
        num_output_tiles,
        mcast_receiver_noc_x,
        mcast_receiver_noc_y,
        mcast_receiver_semaphore_id>(mcast_reciever_base_address, gather_destination_tile_offset_bytes);

    // Wait for mcast to complete and then push back to intermediate_full_cb which will start the second matmul
    wait_for_mcast<intermediate_full_cb, num_tiles_k>(mcast_sender_semaphore_addr_ptr, debug_enabled);

    // Gather after second matmul so that we can mcast full result to all cores
    gather<
        intermediate_pregather_cb,
        intermediate_full_cb,
        num_output_tiles,
        mcast_receiver_noc_x,
        mcast_receiver_noc_y,
        mcast_receiver_semaphore_id>(mcast_reciever_base_address, gather_destination_tile_offset_bytes);

    // Wait for mcast to complete and then push back to intermediate_full_cb which will start the second matmul
    wait_for_mcast<intermediate_full_cb, num_tiles_k>(mcast_sender_semaphore_addr_ptr, debug_enabled);

    // Copy from intermediate_full_cb to out_cb
    constexpr uint32_t number_of_tiles = 1;
    noc_async_write(
        get_read_ptr(intermediate_full_cb),
        get_noc_addr(get_write_ptr(3)),
        get_tile_size(intermediate_full_cb) * num_tiles_k * number_of_tiles);
    noc_async_write_barrier();
    cb_pop_front(intermediate_full_cb, num_tiles_k);
    cb_push_back(3, num_tiles_k);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include <tools/profiler/kernel_profiler.hpp>

template <uint32_t CbOut, uint32_t NumTiles>
FORCE_INLINE void wait_for_mcast(volatile tt_l1_ptr uint32_t* mcast_sender_semaphore_addr_ptr) {
    DeviceZoneScopedN("mcast_reader_wait_for_mcast");
    cb_reserve_back(CbOut, NumTiles);
    noc_semaphore_wait(mcast_sender_semaphore_addr_ptr, VALID);
    cb_push_back(CbOut, NumTiles);
    noc_semaphore_set(mcast_sender_semaphore_addr_ptr, 0);
}

template <
    uint32_t CbIn,
    uint32_t CbOut,
    uint32_t NumTiles,
    uint32_t MCastReceiverNocX,
    uint32_t MCastReceiverNocY,
    uint32_t MCastReceiverSemaphoreId>
void gather(uint32_t receiver_data_addr, uint32_t offset) {
    DeviceZoneScopedN("gather");
    cb_wait_front(CbIn, NumTiles);

    // Gather to receiver core (write and then signal using semaphore)
    const uint64_t mcast_receiver_noc_coord = get_noc_addr(MCastReceiverNocX, MCastReceiverNocY, 0);
    const uint64_t mcast_receiver_noc_addr = mcast_receiver_noc_coord | (uint64_t)(receiver_data_addr + offset);
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(MCastReceiverSemaphoreId);
    const uint64_t mcast_receiver_semaphore_noc_addr =
        mcast_receiver_noc_coord | (uint64_t)mcast_receiver_semaphore_addr;

    noc_async_write_one_packet<true, true>(get_read_ptr(CbIn), mcast_receiver_noc_addr, get_tile_size(CbIn) * NumTiles);
    noc_semaphore_inc<true>(mcast_receiver_semaphore_noc_addr, 1);
    noc_async_posted_writes_flushed();

    // cb_push_back(CbOut, NumTiles); Don't pop yet because mcast needs to happen first
    cb_pop_front(CbIn, NumTiles);
}

void kernel_main() {
    constexpr uint32_t mm1_full_cb = get_compile_time_arg_val(0);
    constexpr uint32_t weight0_cb = get_compile_time_arg_val(1);
    constexpr uint32_t weight1_cb = get_compile_time_arg_val(2);
    constexpr uint32_t intermediate_pregather_cb = get_compile_time_arg_val(3);
    constexpr uint32_t mm2_full_cb = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(6);

    constexpr uint32_t mcast_receiver_noc_x = get_compile_time_arg_val(7);
    constexpr uint32_t mcast_receiver_noc_y = get_compile_time_arg_val(8);
    constexpr uint32_t mcast_receiver_semaphore_id = get_compile_time_arg_val(9);
    constexpr uint32_t mcast_receiver_cb = get_compile_time_arg_val(10);

    const uint32_t mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(11));
    volatile tt_l1_ptr uint32_t* mcast_sender_semaphore_addr_ptr =
        (volatile tt_l1_ptr uint32_t*)mcast_sender_semaphore_addr;

    constexpr uint32_t num_layers = get_compile_time_arg_val(12);

    const uint32_t tile_index = get_arg_val<uint32_t>(0);

    const uint32_t mcast_receiver_base_address = get_write_ptr(mcast_receiver_cb);

    constexpr uint32_t num_output_tiles = 1;  // Set to 1 because we only iterate over K tiles at a time

    cb_reserve_back(mm1_full_cb, num_tiles_k);
    cb_push_back(mm1_full_cb, num_tiles_k);

    // Push full stacked weights for all layers: num_tiles_k * num_layers
    // Each layer will pop num_tiles_k tiles as it processes
    constexpr uint32_t total_weight_tiles = num_tiles_k * num_layers;
    cb_reserve_back(weight0_cb, total_weight_tiles);
    cb_push_back(weight0_cb, total_weight_tiles);

    cb_reserve_back(weight1_cb, total_weight_tiles);
    cb_push_back(weight1_cb, total_weight_tiles);

    // Compute gather destination tile offset bytes from runtime tile index
    const uint32_t gather_destination_tile_offset_bytes = tile_index * get_tile_size(intermediate_pregather_cb);

    for (uint32_t layer = 0; layer < num_layers; layer++) {
        {
            DeviceZoneScopedN("layer_gather_and_mcast");

            // Gather after first matmul so that we can mcast full result to all cores
            gather<
                intermediate_pregather_cb,
                mm2_full_cb,
                num_output_tiles,
                mcast_receiver_noc_x,
                mcast_receiver_noc_y,
                mcast_receiver_semaphore_id>(mcast_receiver_base_address, gather_destination_tile_offset_bytes);
            // Wait for mcast to complete and then push back to mm2_full_cb which will start the second matmul
            wait_for_mcast<mm2_full_cb, num_tiles_k>(mcast_sender_semaphore_addr_ptr);
        }
        {
            DeviceZoneScopedN("layer_gather_and_mcast_2");
            // Gather after second matmul so that we can mcast full result to all cores
            gather<
                intermediate_pregather_cb,
                mm2_full_cb,
                num_output_tiles,
                mcast_receiver_noc_x,
                mcast_receiver_noc_y,
                mcast_receiver_semaphore_id>(mcast_receiver_base_address, gather_destination_tile_offset_bytes);
            // Wait for mcast to complete and then push back to mm1_full_cb (ping-pong back)
            wait_for_mcast<mm1_full_cb, num_tiles_k>(mcast_sender_semaphore_addr_ptr);
        }
    }

    {
        DeviceZoneScopedN("copy_output");

        // Calculate offset for this core's data in mm1_full_cb (mcast contains data from all cores)
        const uint32_t src_addr = get_read_ptr(mm1_full_cb) + gather_destination_tile_offset_bytes;
        constexpr uint32_t number_of_tiles_to_copy_into_output = 1;
        noc_async_write(
            src_addr,
            get_noc_addr(get_write_ptr(out_cb)),
            get_tile_size(mm1_full_cb) * number_of_tiles_to_copy_into_output);
        noc_async_write_barrier();
        cb_pop_front(mm1_full_cb, num_tiles_k);
        cb_push_back(out_cb, num_output_tiles);
    }
}

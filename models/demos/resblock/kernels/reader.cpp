// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"

template <
    uint32_t CbIn,
    uint32_t CbOut,
    uint32_t NumTiles,
    uint32_t MCastReceiverNocX,
    uint32_t MCastReceiverNocY,
    uint32_t MCastReceiverSemaphoreId>
void gather(uint32_t receiver_data_addr, uint32_t offset) {
    cb_wait_front(CbIn, NumTiles);
    cb_reserve_back(CbOut, NumTiles);

    // Write all tiles from CbIn to CbOut using noc_async_write (will remove once the gather+mcast is implemented)
    noc_async_write(get_read_ptr(CbIn), get_noc_addr(get_write_ptr(CbOut)), get_tile_size(CbIn) * NumTiles);

    // Gather to receiver core (write and then signal using semaphore)
    const uint64_t mcast_reciever_noc_coord = get_noc_addr(MCastReceiverNocX, MCastReceiverNocY, 0);
    const uint64_t mcast_reciever_noc_addr = mcast_reciever_noc_coord | (uint64_t)(receiver_data_addr + offset);
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(MCastReceiverSemaphoreId);
    const uint64_t mcast_receiver_semaphore_noc_addr =
        mcast_reciever_noc_coord | (uint64_t)mcast_receiver_semaphore_addr;

    noc_async_write_one_packet<true, true>(get_read_ptr(CbIn), mcast_reciever_noc_addr, get_tile_size(CbIn) * NumTiles);
    noc_semaphore_inc<true>(mcast_receiver_semaphore_noc_addr, 1);
    noc_async_posted_writes_flushed();

    cb_push_back(CbOut, NumTiles);
    cb_pop_front(CbIn, NumTiles);
}

void kernel_main() {
    constexpr uint32_t in0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t weight0_cb = get_compile_time_arg_val(1);
    constexpr uint32_t weight1_cb = get_compile_time_arg_val(2);
    constexpr uint32_t interm_cb = get_compile_time_arg_val(3);
    constexpr uint32_t interm_cb2 = get_compile_time_arg_val(4);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(5);

    constexpr uint32_t mcast_receiver_noc_x = get_compile_time_arg_val(6);
    constexpr uint32_t mcast_receiver_noc_y = get_compile_time_arg_val(7);
    constexpr uint32_t mcast_receiver_semaphore_id = get_compile_time_arg_val(8);
    constexpr uint32_t mcast_reciever_cb = get_compile_time_arg_val(9);

    const uint32_t mcast_reciever_base_address = get_write_ptr(mcast_reciever_cb);

    cb_reserve_back(in0_cb, num_tiles_k);
    cb_push_back(in0_cb, num_tiles_k);

    cb_reserve_back(weight0_cb, num_tiles_k);
    cb_push_back(weight0_cb, num_tiles_k);

    cb_reserve_back(weight1_cb, num_tiles_k);
    cb_push_back(weight1_cb, num_tiles_k);

    // Gather after first matmul so that we can mcast full result to all cores
    constexpr uint32_t num_output_tiles =
        1;  // This is always one because we only iterate over K tiles at a time for now
    gather<
        interm_cb,
        interm_cb2,
        num_output_tiles,
        mcast_receiver_noc_x,
        mcast_receiver_noc_y,
        mcast_receiver_semaphore_id>(mcast_reciever_base_address, 0);  // TODO: Add correct offset/addr

    DPRINT << "Reader kernel finished" << ENDL();
}

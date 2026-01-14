// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "api/debug/dprint.h"

template <
    uint32_t CbIn,
    uint32_t CbOut,
    uint32_t NumTiles,
    uint32_t ReceiverNocX,
    uint32_t ReceiverNocY,
    uint32_t ReceiverSemaphoreId>
void mcast(uint32_t receiver_data_addr, uint32_t offset) {
    cb_wait_front(CbIn, NumTiles);
    cb_reserve_back(CbOut, NumTiles);

    // Write all tiles from CbIn to CbOut using noc_async_write
    noc_async_write(get_read_ptr(CbIn), get_noc_addr(get_write_ptr(CbOut)), get_tile_size(CbIn) * NumTiles);

    // Also gather to receiver core (write and then signal using semaphore)
    const uint64_t reciever_noc_coord = get_noc_addr(ReceiverNocX, ReceiverNocY, 0);
    const uint64_t reciever_noc_addr = reciever_noc_coord | (uint64_t)(receiver_data_addr + offset);
    const uint32_t receiver_semaphore_addr = get_semaphore(ReceiverSemaphoreId);
    const uint64_t receiver_semaphore_noc_addr = reciever_noc_coord | (uint64_t)receiver_semaphore_addr;

    // TODO: Actually do the write, skipping for now
    // noc_async_write_one_packet<true, true>(, dst_data_noc_addr, data_size_bytes);

    noc_semaphore_inc<true>(receiver_semaphore_noc_addr, 1);
    noc_async_write_barrier();

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

    constexpr uint32_t receiver_noc_x = get_compile_time_arg_val(6);
    constexpr uint32_t receiver_noc_y = get_compile_time_arg_val(7);
    constexpr uint32_t receiver_semaphore_id = get_compile_time_arg_val(8);

    DPRINT << "Reader kernel started" << ENDL();
    DPRINT << "In0 CB: " << in0_cb << ENDL();
    DPRINT << "Weight0 CB: " << weight0_cb << ENDL();
    DPRINT << "Weight1 CB: " << weight1_cb << ENDL();
    DPRINT << "Interm CB: " << interm_cb << ENDL();
    DPRINT << "Interm CB2: " << interm_cb2 << ENDL();
    DPRINT << "Num Tiles K: " << num_tiles_k << ENDL();
    DPRINT << "Receiver NOC X: " << receiver_noc_x << ENDL();
    DPRINT << "Receiver NOC Y: " << receiver_noc_y << ENDL();
    DPRINT << "Receiver Semaphore ID: " << receiver_semaphore_id << ENDL();

    cb_reserve_back(in0_cb, num_tiles_k);
    cb_push_back(in0_cb, num_tiles_k);

    cb_reserve_back(weight0_cb, num_tiles_k);
    cb_push_back(weight0_cb, num_tiles_k);

    cb_reserve_back(weight1_cb, num_tiles_k);
    cb_push_back(weight1_cb, num_tiles_k);

    // Mcast single output tile from interm_cb to interm_cb2
    constexpr uint32_t num_output_tiles = 1;  // This is always one because we only iterate over K tiles at a time
    mcast<interm_cb, interm_cb2, num_output_tiles, receiver_noc_x, receiver_noc_y, receiver_semaphore_id>(
        0, 0);  // TODO: Add correct offset/addr

    DPRINT << "Reader kernel finished" << ENDL();
}

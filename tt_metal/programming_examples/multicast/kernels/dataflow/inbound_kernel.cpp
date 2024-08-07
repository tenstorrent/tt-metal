// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {

     ////////// RUNTIME ARGS & VARS //////////
    uint32_t start_x = get_arg_val<uint32_t>(0);
    uint32_t start_y = get_arg_val<uint32_t>(1);
    uint32_t sender_addr = get_arg_val<uint32_t>(2);
    uint32_t receiver_addr = get_arg_val<uint32_t>(3);
    uint32_t kernel_start_ac_target_val = 1;

    ////////// BUFFER SETUP //////////
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0; // index=0
    uint32_t ublock_size_bytes = get_tile_size(cb_id_in0);
    uint32_t l1_addr = get_write_ptr(cb_id_in0);

    ////////// SEMAPHORE SETUP //////////
    volatile tt_l1_ptr uint32_t* receiver_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_addr);
    volatile tt_l1_ptr uint32_t* sender_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_addr);
    cb_reserve_back(cb_id_in0, 1);
    noc_semaphore_set(receiver_addr_ptr, INVALID);
    uint64_t remote_sender_semaphore_noc_addr = get_noc_addr(start_x, start_y, sender_addr);

    ////////// NOTIFY COORDINATOR CORE OF READINESS //////////
    noc_semaphore_inc(remote_sender_semaphore_noc_addr, 1);

    ////////// WAIT UNTIL COORDINATOR CORE MULTICASTS TILE TO WORKER CORES //////////
    noc_semaphore_wait(receiver_addr_ptr, VALID);

    ////////// PRINT TILE SLICE //////////
    auto sr = SliceRange{.h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 8};
        DPRINT << TileSlice(cb_id_in0, 0, sr, true, false) << ENDL();

    cb_push_back(cb_id_in0, 1);

    DPRINT << "CORE " << (uint32_t) my_x[0] << "," << (uint32_t) my_y[0] << ": Inbound kernel has processed and acknowledged its tile." << ENDL() << ENDL();
}

// SPDX-FileCopyrightText: © 2025 Ryan Barton
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

// Helper function to copy a tile from one CB to another CB (eg. input CB to output CB) via L1.
inline void copy_tile_between_cb(uint32_t src_addr, uint32_t dst_addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    for (uint32_t i = 0; i < bytes / sizeof(uint32_t); i++) {
        dst[i] = src[i];
    }
}

// Ensure this is set: export TT_METAL_DPRINT_CORES='(0,0)-(3,0)'
void kernel_main() {

    ////////// RUNTIME ARGS & VARS //////////
    uint32_t start_x = get_arg_val<uint32_t>(0);
    uint32_t start_y = get_arg_val<uint32_t>(1);
    uint32_t sender_addr = get_semaphore(get_arg_val<uint32_t>(2));
    uint32_t receiver_addr = get_semaphore(get_arg_val<uint32_t>(3));

    ////////// BUFFER SETUP //////////
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0; // index=0
    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;  // index=16
    uint32_t ublock_size_bytes = get_tile_size(cb_id_in0);
    uint32_t l1_addr_in = get_write_ptr(cb_id_in0);
    uint32_t l1_addr_out = get_write_ptr(cb_id_out0);

    ////////// SEMAPHORE SETUP //////////
    volatile tt_l1_ptr uint32_t* receiver_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_addr);
    volatile tt_l1_ptr uint32_t* sender_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_addr);
    cb_reserve_back(cb_id_in0, 1);
    noc_semaphore_set(receiver_addr_ptr, INVALID);
    uint64_t remote_sender_semaphore_noc_addr = get_noc_addr(start_x, start_y, sender_addr);

    ////////// NOTIFY COORDINATOR CORE OF READINESS //////////
    noc_semaphore_inc(remote_sender_semaphore_noc_addr, 1);

    ////////// WAIT UNTIL COORDINATOR CORE MULTICASTS TILE TO RECEIVER CORES //////////
    noc_semaphore_wait(receiver_addr_ptr, VALID);

    // At this stage, the receiver cores should have now received the tile.

    ////////// PRINT TILE (8-ELEMENT-STRIDED TILE SLICE) //////////
    SliceRange sr = SliceRange{.h0 = static_cast<uint8_t>(0), .h1 = static_cast<uint8_t>(32), .hs = 8, .w0 = 0, .w1 = 32, .ws = 8};
    DPRINT << TileSlice(cb_id_in0, 0, sr, TSLICE_INPUT_CB, TSLICE_WR_PTR, true, false);

    ////////// PRINT TILE (FULL TILE) //////////
    // for (uint8_t r = 0; r < 32; ++r) {
    //     SliceRange sr = SliceRange{.h0 = static_cast<uint8_t>(r), .h1 = static_cast<uint8_t>(r+1), .hs = 1, .w0 = 0,
    //     .w1 = 32, .ws = 1}; DPRINT_DATA0({ DPRINT << TileSlice(cb_id_in0, 0, sr, TSLICE_INPUT_CB, TSLICE_WR_PTR,
    //     true, false); });
    // }

    cb_push_back(cb_id_in0, 1);

    DPRINT << "CORE (" << (uint32_t)get_absolute_logical_x() << "," << (uint32_t)get_absolute_logical_y()
           << "): Inbound kernel has received and acknowledged its tile." << ENDL() << ENDL();

    ////////// COPY TILE TO OUTBOUND KERNEL'S CB //////////
    cb_reserve_back(cb_id_out0, 1);
    copy_tile_between_cb(get_read_ptr(cb_id_in0), get_write_ptr(cb_id_out0), ublock_size_bytes);
    cb_push_back(cb_id_out0, 1);
}

// SPDX-FileCopyrightText: © 2025 Ryan Barton
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"  // required in all kernels using DPRINT
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"
#include "ckernel.h"  // ckernel::load_blocking

// Copy a tile from one dataflow buffer to another via L1, blocking until the tile has landed in L1.
// The caller's push_back() writes a credit to a STREAM register (a different memory region, so no
// store-ordering guarantee vs these L1 stores); without the drain the consumer could see that credit
// and NoC-read stale L1. load_blocking on the last written word stalls until it lands, and since
// same-region stores are processed in order, draining the last drains all.
inline void copy_tile_between_cb(uint32_t src_addr, uint32_t dst_addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    const uint32_t num_words = bytes / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_words; i++) {
        dst[i] = src[i];
    }
    (void)ckernel::load_blocking(&dst[num_words - 1]);
}

// Ensure this is set: export TT_METAL_DPRINT_CORES='(0,0)-(3,0)'
void kernel_main() {
    ////////// RUNTIME ARGS & VARS //////////
    uint32_t start_x = get_arg(args::start_x);
    uint32_t start_y = get_arg(args::start_y);

    ////////// BUFFER SETUP //////////
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_out0(dfb::out0);
    uint32_t ublock_size_bytes = dfb_in0.get_entry_size();
    uint32_t l1_addr_in = dfb_in0.get_write_ptr();
    uint32_t l1_addr_out = dfb_out0.get_write_ptr();

    ////////// SEMAPHORE SETUP //////////
    Semaphore receiver_sem(sem::receiver);
    Semaphore sender_sem(sem::sender);
    Noc noc;
    dfb_in0.reserve_back(1);
    receiver_sem.set(INVALID);

    ////////// NOTIFY COORDINATOR CORE OF READINESS //////////
    sender_sem.up(noc, start_x, start_y, 1);

    ////////// WAIT UNTIL COORDINATOR CORE MULTICASTS TILE TO RECEIVER CORES //////////
    receiver_sem.wait(VALID);

    // At this stage, the receiver cores should have now received the tile.

    ////////// PRINT TILE (8-ELEMENT-STRIDED TILE SLICE) //////////
    SliceRange sr =
        SliceRange{.h0 = static_cast<uint8_t>(0), .h1 = static_cast<uint8_t>(32), .hs = 8, .w0 = 0, .w1 = 32, .ws = 8};
    DPRINT(
        "Tile slice:\n{}\n",
        TileSlice(static_cast<uint8_t>(dfb_in0.get_id()), 0, sr, TSLICE_INPUT_CB, TSLICE_WR_PTR, true, false));

    ////////// PRINT TILE (FULL TILE) //////////
    // for (uint8_t r = 0; r < 32; ++r) {
    //     SliceRange sr = SliceRange{.h0 = static_cast<uint8_t>(r), .h1 = static_cast<uint8_t>(r+1), .hs = 1, .w0 = 0,
    //     .w1 = 32, .ws = 1};
    //     DPRINT_DATA0("{}\n", TileSlice(static_cast<uint8_t>(dfb_in0.get_id()), 0, sr, TSLICE_INPUT_CB,
    //     TSLICE_WR_PTR, true, false));
    // }

    dfb_in0.push_back(1);

    DPRINT(
        "CORE ({},{}): Inbound kernel has received and acknowledged its tile.\n\n",
        get_absolute_logical_x(),
        get_absolute_logical_y());

    ////////// COPY TILE TO OUTBOUND KERNEL'S DATAFLOW BUFFER //////////
    dfb_out0.reserve_back(1);
    copy_tile_between_cb(dfb_in0.get_read_ptr(), dfb_out0.get_write_ptr(), ublock_size_bytes);
    dfb_out0.push_back(1);
}

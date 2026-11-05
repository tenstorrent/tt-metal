// SPDX-FileCopyrightText: © 2025 Ryan Barton
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"  // required in all kernels using DPRINT
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"

// Ensure this is set: export TT_METAL_DPRINT_CORES='(0,0)-(3,0)'
void kernel_main() {

    ////////// RUNTIME ARGS & VARS //////////
    uint32_t start_x = get_arg(args::start_x);
    uint32_t start_y = get_arg(args::start_y);
    uint32_t end_x = get_arg(args::end_x);
    uint32_t end_y = get_arg(args::end_y);
    uint32_t dram_bank_id = get_arg(args::dram_bank_id);
    uint32_t src0_dram = get_arg(args::src0_dram);
    uint32_t single_tile_size = get_arg(args::single_tile_size);
    uint32_t num_dests = get_arg(args::num_dests);

    ////////// BUFFER SETUP //////////
    uint64_t src0_dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, src0_dram);
    DataflowBuffer dfb_in0(dfb::in0);
    uint32_t ublock_size_bytes = dfb_in0.get_entry_size();
    uint32_t tile_l1_addr = dfb_in0.get_write_ptr();

    ////////// READ TILE DRAM->L1 //////////
    noc_async_read(src0_dram_noc_addr, tile_l1_addr, ublock_size_bytes);
    noc_async_read_barrier();

    ////////// PRINT TILE SLICE //////////
    SliceRange sr = SliceRange{.h0 = static_cast<uint8_t>(0), .h1 = static_cast<uint8_t>(32), .hs = 8, .w0 = 0, .w1 = 32, .ws = 8};
    DPRINT(
        "Tile slice:\n{}\n",
        TileSlice(static_cast<uint8_t>(dfb_in0.get_id()), 0, sr, TSLICE_INPUT_CB, TSLICE_WR_PTR, true, false));

    dfb_in0.push_back(1);

    ////////// SEMAPHORE SETUP //////////
    Semaphore sender_sem(sem::sender);
    Semaphore receiver_sem(sem::receiver);
    Noc noc;

    ////////// WAIT FOR WORKER CORES' READINESS //////////
    sender_sem.wait(num_dests);

    ////////// RESET SEMAPHORE //////////
    sender_sem.set(0);  // set to 0 rather than INVALID because sender_sem can take on a value > 1

    DPRINT(
        "CORE ({},{}): Tile ready for multicast. I am starting all inbound kernels in cores in given range.\n\n",
        get_absolute_logical_x(),
        get_absolute_logical_y());

    ////////// MULTICAST TILE TO RECEIVERS //////////
    uint64_t identity_tile_global_multicast_addr =
            get_noc_multicast_addr(start_x, start_y, end_x, end_y, tile_l1_addr);
    noc_async_write_multicast(tile_l1_addr, identity_tile_global_multicast_addr, single_tile_size, num_dests);

    ////////// MULTICAST 'VALID' STATE TO RECEIVERS //////////
    receiver_sem.set(VALID);
    receiver_sem.set_multicast(noc, start_x, start_y, end_x, end_y, num_dests);

    noc_async_write_barrier();

    // From this stage onwards, the coordinator can safely do more work.
    // The user is encouraged to play around with coordinator behavior (i.e., retrieve more tiles from DRAM, aggregate tile work from worker cores, etc.) as an exercise.
}

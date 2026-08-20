// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Ring rotation with hand-built cross-core flow control. Core i reads its tiles from DRAM into a
// private scratchpad and NoC-writes each one straight into core (i+1)'s `recv` DFB entry, while
// consuming what core (i-1) sends it.
//
// Two things this kernel has to do by hand, because Metal 2.0's DFB is node-local:
//
//   1. ADDRESSING. The destination is a UnicastEndpoint whose `addr` is *my own*
//      dfb_recv.get_write_ptr(). There is no name for "the recv DFB on core (x,y)". This only works
//      because a DFB is allocated identically on every node AND both sides are at the same loop
//      iteration, so their write pointers sit in the same ring slot. Nothing checks either premise.
//
//   2. BACK-PRESSURE. reserve_back()/push_back() flow control stops at the node boundary: my
//      push_back tells my LOCAL writer, not my remote producer. So the `space` semaphore is a
//      hand-rolled reverse credit -- I tell my PREDECESSOR that I have reserved a slot, and I wait
//      for my SUCCESSOR to say the same before writing. Without it a fast core overruns a slow
//      one's ring slot and silently corrupts a tile.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "api/scratchpad.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_page = get_arg(args::first_page);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t next_x = get_arg(args::next_x);
    const uint32_t next_y = get_arg(args::next_y);
    const uint32_t prev_x = get_arg(args::prev_x);
    const uint32_t prev_y = get_arg(args::prev_y);
    // Compile-time knob: with 0, the hand-rolled reverse credit is gone and a fast core is free to
    // overrun a slow one's ring slot. Nothing in the DFB stops it -- flow control is node-local.
    constexpr uint32_t use_credit = get_arg(args::use_credit);

    Noc noc;
    DataflowBuffer dfb_recv(dfb::recv);
    Scratchpad<uint32_t> stage(scratch::stage);
    const auto acc_in = TensorAccessor(tensor::in);
    Semaphore arrived(sem::arrived);
    Semaphore space(sem::space);
    UnicastEndpoint peer;

    const uint32_t tile_bytes = dfb_recv.get_tile_size();

    for (uint32_t t = 0; t < num_tiles; ++t) {
        // 1. my tile -> my private scratchpad
        noc.async_read(acc_in, stage, tile_bytes, {.page_id = first_page + t}, {.offset_bytes = 0});
        noc.async_read_barrier();

        // 2. claim my own recv slot. This address doubles as the REMOTE address in step 4.
        dfb_recv.reserve_back(1);
        const uint32_t entry = dfb_recv.get_write_ptr();

        // 3. reverse credit: my slot is claimed, predecessor may fill it; wait for my successor's.
        if constexpr (use_credit) {
            space.up(noc, prev_x, prev_y, 1);
            space.down(1);
        }

        // 4. scratchpad -> successor's recv slot, then tell it the tile landed.
        noc.async_write(
            stage, peer, tile_bytes, {.offset_bytes = 0}, {.noc_x = next_x, .noc_y = next_y, .addr = entry});
        noc.async_write_barrier();
        arrived.up(noc, next_x, next_y, 1);

        // 5. wait for my predecessor, then hand the slot to the writer. I am declared this DFB's
        //    PRODUCER but I never wrote a byte of it -- a remote core did.
        arrived.down(1);
        dfb_recv.push_back(1);
    }
}

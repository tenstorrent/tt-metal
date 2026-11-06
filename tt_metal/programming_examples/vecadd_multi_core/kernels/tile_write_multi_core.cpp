// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t n_tiles = get_arg(args::n_tiles);
    uint32_t start_tile_id = get_arg(args::start_tile_id);

    // The dataflow buffer that we are going to read from and write to DRAM
    DataflowBuffer dfb_out0(dfb::out0);

    // Address generator for the output buffer. This is faster than doing plain
    // DRAM writes.
    // The output tensor is bound to this kernel by name in the host code.
    const auto c = TensorAccessor(tensor::c);

    // Calculate the range of tiles this core should process
    const uint32_t end_tile_id = start_tile_id + n_tiles;

    // Loop over the assigned tiles and write them to the output buffer
    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        // Make sure there is a tile in the dataflow buffer
        dfb_out0.wait_front(1);
        uint32_t dfb_out0_addr = dfb_out0.get_read_ptr();
        // write the tile to DRAM
        noc_async_write_page(i, c, dfb_out0_addr);
        // This will wait until the write is done. As an alternative, noc_async_writes_flushed()
        // can be faster because it waits until the write request is sent. In that case, you
        // have to use noc_async_write_barrier() at least once at the end of data movement
        // kernel to make sure all writes are done.
        noc_async_write_barrier();
        // Mark the tile as consumed
        dfb_out0.pop_front(1);
    }
}

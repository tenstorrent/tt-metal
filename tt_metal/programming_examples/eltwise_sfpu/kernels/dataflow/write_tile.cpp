// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t n_tiles = get_arg(args::n_tiles);

    // The dataflow buffer that we are going to read from and write to DRAM
    DataflowBuffer dfb_out(dfb::out0);
    const uint32_t tile_size_bytes = dfb_out.get_entry_size();

    // The output tensor is bound to this kernel by name in the host code.
    const auto out0 = TensorAccessor(tensor::out0);

    // Create Device 2.0 Noc object
    Noc noc;

    // Loop over all the tiles and write them to the output buffer
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Make sure there is a tile in the dataflow buffer
        dfb_out.wait_front(1);
        // Write the tile to DRAM
        noc.async_write(dfb_out, out0, tile_size_bytes, {}, {.page_id = i});
        noc.async_write_barrier();  // This will wait until the write is done. As an alternative,
                                    // noc_async_write_flushed() can be faster because it waits
                                    // until the write request is sent. In that case, you have to
                                    // use noc_async_write_barrier() at least once at the end of
                                    // data movement kernel to make sure all writes are done.
        // Mark the tile as consumed
        dfb_out.pop_front(1);
    }
}

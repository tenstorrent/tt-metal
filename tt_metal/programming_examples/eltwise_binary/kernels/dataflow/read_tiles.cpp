// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t n_tiles = get_arg(args::n_tiles);

    // The dataflow buffers to read the tiles into
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    // Get the tile size used in the dataflow buffers. We assume the
    // dataflow buffers are created with the same tile size as the DRAM
    // buffers (This is most of the cases).
    const uint32_t tile_size_bytes = dfb_in0.get_entry_size();

    // Create address generators for the input buffers. Consider these the
    // pointers for interleaved buffers
    // Setting the page size to be tile_size_bytes works because we set it up
    // explicitly in host code. This is usually a good idea as it makes coding
    // easy.
    // The input tensors are bound to this kernel by name in the host code.
    const auto in0 = TensorAccessor(tensor::in0);
    const auto in1 = TensorAccessor(tensor::in1);

    // Create Device 2.0 Noc object
    Noc noc;

    // Loop over all the tiles and read them into the dataflow buffers
    for (uint32_t i = 0; i < n_tiles; i++) {
        // First make sure there is space in the dataflow buffers to be written to.
        dfb_in0.reserve_back(1);
        dfb_in1.reserve_back(1);  // wait until we have 1 free slot. This blocks if the
                                  // other kernels cannot consume the tiles fast enough.
                                  // Deciding how large the buffer should be is a tradeoff.
        // read the tile into the dataflow buffer
        noc.async_read(in0, dfb_in0, tile_size_bytes, {.page_id = i}, {.offset_bytes = 0});
        // We can overlap async reads and writes to reduce the data movement overhead.
        noc.async_read(in1, dfb_in1, tile_size_bytes, {.page_id = i}, {.offset_bytes = 0});

        noc.async_read_barrier();  // Wait until tile reads are done
        dfb_in0.push_back(1);
        dfb_in1.push_back(1);  // mark the tiles as ready. From this point forward kernels
                               // calling `wait_front` will see this tile
    }
}

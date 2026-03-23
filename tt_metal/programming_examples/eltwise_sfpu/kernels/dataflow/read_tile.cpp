// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    // The circular buffers to read the tiles into
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;

    // Get the tile size used in the circular buffers. We assume the
    // circular buffers are created with the same tile size as the DRAM
    // buffers (This is most of the cases).
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // Create address generators for the input buffers. Consider these the
    // pointers for interleaved buffers
    // Setting the page size to be tile_size_bytes works because we set it up
    // explicitly in host code. This is usually a good idea as it makes coding
    // easy.
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_size_bytes);

    // Create Device 2.0 experimental Noc and CircularBuffer objects
    experimental::Noc noc;
    experimental::CircularBuffer cb_in0_buf(cb_in0);

    // Loop over all the tiles and read them into the circular buffers
    for (uint32_t i = 0; i < n_tiles; i++) {
        // First make sure there is space in the circular buffers to be written to.
        cb_in0_buf.reserve_back(1);
        // read the tile into the circular buffer
        // We can overlap async reads and writes to reduce the data movement overhead.
        noc.async_read(in0, cb_in0_buf, tile_size_bytes, {.page_id = i}, {.offset_bytes = 0});

        noc.async_read_barrier();  // Wait until tile reads are done
        cb_in0_buf.push_back(1);   // mark the tiles as ready. From this point forward kernels
                                   // calling `cb_wait_front` will see this tile
    }
}

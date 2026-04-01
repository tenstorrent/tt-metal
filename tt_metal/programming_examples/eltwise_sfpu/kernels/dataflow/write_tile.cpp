// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t c_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    // The circular buffer that we are going to read from and write to DRAM
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    // Address of the output buffer
    constexpr auto out0_args = TensorAccessorArgs<0>();
    const auto out0 = TensorAccessor(out0_args, c_addr, tile_size_bytes);

    // Create Device 2.0 experimental Noc and CircularBuffer objects
    experimental::Noc noc;
    experimental::CircularBuffer cb_out(cb_out0);

    // Loop over all the tiles and write them to the output buffer
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Make sure there is a tile in the circular buffer
        cb_out.wait_front(1);
        // Write the tile to DRAM
        noc.async_write(cb_out, out0, tile_size_bytes, {}, {.page_id = i});
        noc.async_write_barrier();  // This will wait until the write is done. As an alternative,
                                    // noc_async_write_flushed() can be faster because it waits
                                    // until the write request is sent. In that case, you have to
                                    // use noc_async_write_barrier() at least once at the end of
                                    // data movement kernel to make sure all writes are done.
        // Mark the tile as consumed
        cb_out.pop_front(1);
    }
}

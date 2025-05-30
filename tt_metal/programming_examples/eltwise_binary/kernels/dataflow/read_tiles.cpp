// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_addr = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);

    // The circular buffers to read the tiles into
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    // Get the tile size used in the circular buffers. We assume the
    // circular buffers are created with the same tile size as the DRAM
    // buffers (Whis is most of the cases).
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // Create address generators for the input buffers. Consider these the
    // pointers for interleaved buffers
    // Setting the page size to be tile_size_bytes works because we set it up
    // explicitly in host code. This is usually a good idea as it makes coding
    // easy.
    const InterleavedAddrGenFast<true> in0 = {
        .bank_base_address = in0_addr,         // The base address of the buffer
        .page_size = tile_size_bytes,          // The size of a buffer page
        .data_format = DataFormat::Float16_b,  // The data format of the buffer
    };
    const InterleavedAddrGenFast<true> in1 = {
        .bank_base_address = in1_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    // Loop over all the tiles and read them into the circular buffers
    for (uint32_t i = 0; i < n_tiles; i++) {
        // First make sure there is space in the circular buffers to be written to.
        cb_reserve_back(cb_in0, 1);
        cb_reserve_back(cb_in1, 1);  // wait until we have 1 free slot. This blocks if the
                                     // other kernels cannot consume the tiles fast enough.
                                     // Deciding how large the buffer should be is a tradeoff.
        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
        uint32_t cb_in1_addr = get_write_ptr(cb_in1);
        noc_async_read_tile(i, in0, cb_in0_addr);  // read the tile into the circular buffer
        noc_async_read_tile(i, in1, cb_in1_addr);  // We can overlap async reads and writes
                                                   // to reduce the data movement overhead.

        noc_async_read_barrier();  // Wait until tile reads are done
        cb_push_back(cb_in0, 1);
        cb_push_back(cb_in1, 1);  // mark the tiles as ready. From this point forward kernels
                                  // calling `cb_wait_front` will see this tile
    }
}

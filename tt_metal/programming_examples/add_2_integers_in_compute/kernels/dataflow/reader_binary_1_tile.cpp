// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_addr = get_arg_val<uint32_t>(1);

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

    // read the tiles from DRAM into the circular buffers
    cb_reserve_back(cb_in0, 1);
    uint32_t cb_in0_addr = get_write_ptr(cb_in0);
    noc_async_read_tile(0, in0, cb_in0_addr);  // read
    noc_async_read_barrier();                  // wait until the read is done
    cb_push_back(cb_in0, 1);                   // mark the tile as ready.

    // same process for the second input (different circular buffer and input buffer)
    cb_reserve_back(cb_in1, 1);
    uint32_t cb_in1_addr = get_write_ptr(cb_in1);
    noc_async_read_tile(0, in1, cb_in1_addr);
    noc_async_read_barrier();
    cb_push_back(cb_in1, 1);
}

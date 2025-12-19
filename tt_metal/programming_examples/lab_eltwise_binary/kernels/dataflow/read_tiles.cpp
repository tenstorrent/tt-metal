// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t in0_base_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_base_addr = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);

    // The circular buffers to read the tiles into
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    // Get the tile size used in the circular buffers. We assume the
    // circular buffers are created with the same tile size as the DRAM
    // buffers (Whis is most of the cases).
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // Create address generators for the input buffers. Address generators can determine
    // physical address based on the provided data layout and base address.
    // Observe that here we are just constructing the address generators, but not using them yet.
    // They are used in the loop below to read the tiles from source memory into the circular buffers.
    constexpr auto in0_layout_args = TensorAccessorArgs<0>();
    const auto in0_addr_gen = TensorAccessor(in0_layout_args, in0_base_addr, tile_size_bytes);
    constexpr auto in1_layout_args = TensorAccessorArgs<in0_layout_args.next_compile_time_args_offset()>();
    const auto in1_addr_gen = TensorAccessor(in1_layout_args, in1_base_addr, tile_size_bytes);

    // Loop over all the tiles and read them into the circular buffers
    for (uint32_t i = 0; i < n_tiles; i++) {
        // First make sure there is space in the circular buffers to be written to.
        cb_reserve_back(cb_in0, 1);
        cb_reserve_back(cb_in1, 1);  // Wait until we have 1 free slot. This blocks if the
                                     // other kernels cannot consume the tiles fast enough.
                                     // Deciding how large the buffer should be is a tradeoff.
        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
        uint32_t cb_in1_addr = get_write_ptr(cb_in1);
        noc_async_read_tile(i, in0_addr_gen, cb_in0_addr);  // read the tile into the circular buffer
        noc_async_read_tile(i, in1_addr_gen, cb_in1_addr);  // We can overlap async reads and writes
                                                            // to reduce the data movement overhead.
        // Wait until both reads are done before signaling the circular buffer that the tiles are ready.
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
        cb_push_back(cb_in1, 1);  // mark the tiles as ready. From this point forward kernels
                                  // calling `cb_wait_front` will see this tile
    }
}

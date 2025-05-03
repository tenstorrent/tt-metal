// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include <cstdint>

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);
    uint32_t start_tile_id = get_arg_val<uint32_t>(3);

    // The circular buffers to read the tiles into
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);

    // Get the tile size used in the circular buffers. We assume the
    // circular buffers are created with the same tile size as the DRAM
    // buffers. (Whis is most of the cases)
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);
    // DPRINT_DATA0(DPRINT << "tile_size_bytes " <<  tile_size_bytes << ENDL());
    //  Create address generators for the input buffers. This is much faster
    //  then doing plain DRAM reads.
    //  Setting the page size to be tile_size_bytes works because we set it up
    //  explicitly in host code. This is usually a good idea as it makes coding
    //  easy. But may not be the most efficient way to do it in all cases.
    const InterleavedAddrGenFast<true> a = {
        .bank_base_address = a_addr,           // The base address of the buffer
        .page_size = tile_size_bytes,          // The size of a buffer page
        .data_format = DataFormat::Float16_b,  // The data format of the buffer
    };
    const InterleavedAddrGenFast<true> b = {
        .bank_base_address = b_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    // Calculate the range of tiles this core should process
    const uint32_t end_tile_id = start_tile_id + n_tiles;

    // Now we loop over the assigned tiles and read them into the circular
    // buffers
    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        // First we make sure there is space in the circular buffers
        cb_reserve_back(cb_in0, 1);
        cb_reserve_back(
            cb_in1,
            1);  // wait until we have 1 free slot. This blocks if the
                 // other kernels cannot consume the tiles fast enough.
                 // Deciding how large the buffer should be is a tradeoff.
        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
        uint32_t cb_in1_addr = get_write_ptr(cb_in1);
        noc_async_read_tile(i, a, cb_in0_addr);  // read the tile into the circular buffer
        noc_async_read_tile(i, b, cb_in1_addr);  // We can overlap async reads and writes
                                                 // to reduce the data movement overhead.

        // NOTE: Since circular buffers are backed by SRAM, we can actually
        // access them by casting the address to a pointer. This is not helpful
        // in most cases as the CPU is quite slow compared to the tensor/simd
        // engines. But useful for debugging. uint16_t* ptr =
        // (uint16_t*)cb_in0_addr; DPRINT << "cb_in0_addr: " << ptr << " " <<
        // *ptr;

        noc_async_read_barrier();  // Wait until tile reads are done
        cb_push_back(cb_in0, 1);
        cb_push_back(
            cb_in1,
            1);  // mark the tiles as ready. From this point forward
                 // kernels calling `cb_wait_front` will see this tile
    }
}

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"
#include <cstdint>

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t n_tiles = get_arg(args::n_tiles);
    uint32_t start_tile_id = get_arg(args::start_tile_id);

    // The dataflow buffers to read the tiles into
    // These buffers are accessed by the names bound to this kernel in the
    // kernel spec.
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    // Get the tile size used in the dataflow buffers. We assume the
    // dataflow buffers are created with the same tile size as the DRAM
    // buffers. (This is most of the cases)
    const uint32_t tile_size_bytes = dfb_in0.get_entry_size();
    // DPRINT_DATA0("tile_size_bytes {}\n", tile_size_bytes);
    //  Create address generators for the input buffers. This is much faster
    //  then doing plain DRAM reads.
    //  Setting the page size to be tile_size_bytes works because we set it up
    //  explicitly in host code. This is usually a good idea as it makes coding
    //  easy. But may not be the most efficient way to do it in all cases.
    //  The input tensors are bound to this kernel by name in the host code.
    const auto a = TensorAccessor(tensor::a);
    const auto b = TensorAccessor(tensor::b);

    // Calculate the range of tiles this core should process
    const uint32_t end_tile_id = start_tile_id + n_tiles;

    // Now we loop over the assigned tiles and read them into the dataflow
    // buffers
    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        // First we make sure there is space in the dataflow buffers
        dfb_in0.reserve_back(1);
        dfb_in1.reserve_back(1);  // wait until we have 1 free slot. This blocks if the
                                  // other kernels cannot consume the tiles fast enough.
                                  // Deciding how large the buffer should be is a tradeoff.
        uint32_t dfb_in0_addr = dfb_in0.get_write_ptr();
        uint32_t dfb_in1_addr = dfb_in1.get_write_ptr();
        noc_async_read_page(i, a, dfb_in0_addr);  // read the tile into the dataflow buffer
        noc_async_read_page(i, b, dfb_in1_addr);  // We can overlap async reads and writes
                                                  // to reduce the data movement overhead.

        // NOTE: Since dataflow buffers are backed by SRAM, we can actually
        // access them by casting the address to a pointer. This is not helpful
        // in most cases as the CPU is quite slow compared to the tensor/simd
        // engines. But useful for debugging.
        // uint16_t* ptr = (uint16_t*)dfb_in0_addr;
        // DPRINT("dfb_in0_addr: {} {}\n", ptr, *ptr);

        noc_async_read_barrier();  // Wait until tile reads are done
        dfb_in0.push_back(1);
        dfb_in1.push_back(1);  // mark the tiles as ready. From this point forward
                               // kernels calling `wait_front` will see this tile
    }
}

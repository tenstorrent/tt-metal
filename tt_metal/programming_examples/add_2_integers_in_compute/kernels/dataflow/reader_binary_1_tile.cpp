// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t in0_addr = get_arg(args::in0_addr);
    uint32_t in1_addr = get_arg(args::in1_addr);

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

    // read the tiles from DRAM into the dataflow buffers
    dfb_in0.reserve_back(1);
    uint32_t dfb_in0_addr = dfb_in0.get_write_ptr();
    noc_async_read_page(0, in0, dfb_in0_addr);  // read
    noc_async_read_barrier();                   // wait until the read is done
    dfb_in0.push_back(1);                       // mark the tile as ready.

    // same process for the second input (different dataflow buffer and input buffer)
    dfb_in1.reserve_back(1);
    uint32_t dfb_in1_addr = dfb_in1.get_write_ptr();
    noc_async_read_page(0, in1, dfb_in1_addr);
    noc_async_read_barrier();
    dfb_in1.push_back(1);
}

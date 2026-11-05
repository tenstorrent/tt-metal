// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Runtime arguments to write data back into the output tensor.
    uint32_t num_tiles = get_arg(args::num_tiles);  // number of output tiles to write
    uint32_t start_id = get_arg(args::start_id);    // starting tile ID for output tiles

    DataflowBuffer dfb_out(dfb::out0);

    // Create the address generator for the output tensor. Due to us sharing tensor and dataflow buffer
    // configuration parameters (e.g. same data type and same page size) in the host code, we can grab
    // the same parameters from the dataflow buffer as we would from the DRAM tensor.
    constexpr uint32_t onetile = 1;  // single-tile ublocks

    // The output tensor is bound to this kernel by name in the host code.
    const auto c = TensorAccessor(tensor::c);

    // Loop through the tile indices and write each tile to DRAM in order.
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        // Wait for the kernel to produce an output tile
        dfb_out.wait_front(onetile);
        // Write the output tile to DRAM.
        uint32_t l1_read_addr = dfb_out.get_read_ptr();
        noc_async_write_page(i, c, l1_read_addr);
        noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                    // noc_async_write_flushed() can be faster because it waits
                                    // until the write request is sent. In that case, you have to
                                    // use noc_async_write_barrier() at least once at the end of
                                    // data movement kernel to make sure all writes are done.
        dfb_out.pop_front(onetile);
    }
}

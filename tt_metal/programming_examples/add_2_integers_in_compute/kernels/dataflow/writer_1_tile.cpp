// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t dst_addr = get_arg(args::dst_addr);

    // The dataflow buffer that we are going to read from and write to DRAM
    DataflowBuffer dfb_out0(dfb::out0);
    const uint32_t tile_size_bytes = dfb_out0.get_entry_size();

    // Address of the output buffer
    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    // Make sure there is a tile in the dataflow buffer
    dfb_out0.wait_front(1);
    uint32_t dfb_out0_addr = dfb_out0.get_read_ptr();
    // write the tile to DRAM
    noc_async_write_page(0, dst, dfb_out0_addr);
    noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                // noc_async_write_flushed() can be faster because it waits
                                // until the write request is sent. In that case, you have to
                                // use noc_async_write_barrier() at least once at the end of
                                // data movement kernel to make sure all writes are done.
    // Mark the tile as consumed
    dfb_out0.pop_front(1);
}

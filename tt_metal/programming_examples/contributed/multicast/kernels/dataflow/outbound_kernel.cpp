// SPDX-FileCopyrightText: © 2025 Ryan Barton
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    ////////// RUNTIME ARGS & VARS //////////
    uint32_t tile_offset = get_arg(args::tile_offset);

    ////////// BUFFER SETUP //////////
    DataflowBuffer dfb_out0(dfb::out0);
    // The output tensor is bound to this kernel by name in the host code.
    const auto dram_writer = TensorAccessor(tensor::dst);

    dfb_out0.wait_front(1);
    uint32_t l1_read_addr = dfb_out0.get_read_ptr();

    ////////// CALCULATE OFFSET AND WRITE TO DRAM //////////
    uint32_t dram_tile_id = tile_offset;
    noc_async_write_page(dram_tile_id, dram_writer, l1_read_addr);
    noc_async_write_barrier();

    dfb_out0.pop_front(1);

    DPRINT(
        "Core ({},{}): Outbound kernel has written tile to DRAM index {}.\n",
        get_absolute_logical_x(),
        get_absolute_logical_y(),
        dram_tile_id);
}

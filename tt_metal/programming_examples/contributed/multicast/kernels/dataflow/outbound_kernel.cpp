// SPDX-FileCopyrightText: © 2025 Ryan Barton
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main() {
    ////////// RUNTIME ARGS & VARS //////////
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t tile_offset = get_arg_val<uint32_t>(1);

    ////////// BUFFER SETUP //////////
    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;  // CBIndex::c_16
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);
    const InterleavedAddrGenFast<true> dram_writer = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    cb_wait_front(cb_id_out0, 1);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

    ////////// CALCULATE OFFSET AND WRITE TO DRAM //////////
    uint32_t dram_tile_id = tile_offset;
    noc_async_write_tile(dram_tile_id, dram_writer, l1_read_addr);
    noc_async_write_barrier();

    cb_pop_front(cb_id_out0, 1);

    DPRINT << "Core (" << (uint32_t)get_absolute_logical_x() << "," << (uint32_t)get_absolute_logical_y()
           << "): Outbound kernel has written tile to DRAM index " << dram_tile_id << "." << ENDL();
}

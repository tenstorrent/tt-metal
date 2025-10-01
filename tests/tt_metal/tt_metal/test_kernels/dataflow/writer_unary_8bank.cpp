// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

constexpr bool get_write_to_dram() {
    if constexpr (kernel_compile_time_args.size() > 0) {
        return get_compile_time_arg_val(0);
    } else {
        return true;
    }
}

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(2); // Index 2 to match with regular writer_unary

    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);

    constexpr bool write_to_dram = get_write_to_dram();

    const InterleavedAddrGenFast<write_to_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    // Wait on all output tiles at once (like reduce_c pattern)
    cb_wait_front(cb_id_out0, num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

    for (uint32_t i = 0; i < num_tiles; i++) {
        uint64_t dst_noc_addr = get_noc_addr(i, s);
        uint32_t tile_addr = l1_read_addr + (i * tile_bytes);
        noc_async_write_tile(i, s, tile_addr);
    }

    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, num_tiles);
}

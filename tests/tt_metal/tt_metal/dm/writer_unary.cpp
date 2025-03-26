// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// L1 to DRAM write
// TODO: Expand this to write to other core(s)
void kernel_main() {
    uint32_t dst_addr = get_compile_time_arg_val(0);
    constexpr uint32_t bank_id = get_compile_time_arg_val(1);
    constexpr uint32_t total_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t ublock_size_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(4);

    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0) * ublock_size_tiles;

    for (uint32_t i = 0; i < total_num_tiles; i += ublock_size_tiles) {
        // TODO: Change dst address to change DRAM/core locations (single/multiple core)
        uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dst_addr);

        cb_wait_front(cb_id_out0, ublock_size_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);

        noc_async_write_barrier();

        cb_pop_front(cb_id_out0, ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}

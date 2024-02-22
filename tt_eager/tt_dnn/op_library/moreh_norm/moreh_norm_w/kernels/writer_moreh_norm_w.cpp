// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    int i{0};
    const auto output_addr = get_arg_val<uint32_t>(i++);
    const bool output_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto Wt = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{16};
    const auto cb_id_output = cb_id++;

    const uint32_t output_tile_bytes = get_tile_size(cb_id_output);
    const auto output_data_format = get_dataformat(cb_id_output);

    const InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    const InterleavedAddrGenFast<false> l1_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    const auto start_tile_idx = tile_offset / Wt;
    const auto output_l1_read_addr = get_read_ptr(cb_id_output);

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        const auto tile_idx = start_tile_idx + row_idx;
        cb_wait_front(cb_id_output, 1);
        if (output_is_dram) {
            noc_async_write_tile(tile_idx, dram_output_addrg, output_l1_read_addr);
        } else {
            noc_async_write_tile(tile_idx, l1_output_addrg, output_l1_read_addr);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_output, 1);
    }
}  // void kernel_main()

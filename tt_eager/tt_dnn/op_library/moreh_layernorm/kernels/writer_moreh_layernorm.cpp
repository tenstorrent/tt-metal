// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    const auto output_addr = get_arg_val<uint32_t>(0);
    const auto num_rows_per_core = get_arg_val<uint32_t>(1);
    const auto Wt = get_arg_val<uint32_t>(2);
    const auto tile_offset = get_arg_val<uint32_t>(3);

    constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t block_size = get_compile_time_arg_val(1);

    constexpr uint32_t cb_id_out0 = 16;

    const uint32_t output_tile_bytes = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<output_is_dram> output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = data_format};

    uint32_t offs = 0;
    const auto NCHt = num_rows_per_core;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        for (uint32_t wt = 0; wt < Wt; wt += block_size) {
            cb_wait_front(cb_id_out0, block_size);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
            for (uint32_t r = 0; r < block_size; r++) {
                noc_async_write_tile(offs + wt + r + tile_offset, output_addrg, l1_read_addr);
                l1_read_addr += output_tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_id_out0, block_size);
        }  // wt loop
        offs += Wt;
    }  // ncht loop
}  // void kernel_main()

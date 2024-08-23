// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"


void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(3); // Index 3 to match with regular writer_unary

    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);

    #ifdef KERNEL_COMPILE_TIME_ARG_0
    constexpr bool write_to_dram = get_compile_time_arg_val(0);
    #else
    constexpr bool write_to_dram = true;
    #endif

    const InterleavedAddrGenFast<write_to_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format};

    for (uint32_t i = 0; i<num_tiles; i ++) {
        uint64_t dst_noc_addr = get_noc_addr(i, s);

        cb_wait_front(cb_id_out0, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

        noc_async_write_tile(i, s, l1_read_addr);

        noc_async_write_barrier();

        cb_pop_front(cb_id_out0, onetile);
    }
}

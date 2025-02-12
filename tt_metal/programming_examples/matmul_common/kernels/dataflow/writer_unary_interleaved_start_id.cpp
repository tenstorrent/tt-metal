// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = 4096;
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    cb_wait_front(0, 32);
    uint32_t l1_read_addr = get_read_ptr(0);
    volatile tt_l1_ptr uint32_t* cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_read_addr);
    for (uint32_t i = 0; i < (32 * 4096) / sizeof(uint32_t); ++i) {
        cb_ptr[i] *= 2;
    }

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        noc_async_write_tile(i, s, l1_read_addr);
        noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                    // noc_async_write_flushed() can be faster because it waits
                                    // until the write request is sent. In that case, you have to
                                    // use noc_async_write_barrier() at least once at the end of
                                    // data movement kernel to make sure all writes are done.
        l1_read_addr += 4096;
    }
    cb_pop_front(0, 32);
}

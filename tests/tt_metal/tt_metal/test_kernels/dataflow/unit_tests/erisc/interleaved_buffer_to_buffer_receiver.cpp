// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t tile_bytes = get_arg_val<uint32_t>(3);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_tiles);
#else

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    DataFormat data_format = DataFormat::Float16_b;

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        // cb_wait_front(cb_id_out, onetile);
        uint32_t l1_read_addr = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
        eth_wait_for_bytes(tile_bytes);

        noc_async_write_tile(i, s, l1_read_addr);
        noc_async_write_barrier();

        eth_receiver_done();
    }
#endif
}

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

//#include "debug/dprint.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t tile_bytes = get_arg_val<uint32_t>(3);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    DataFormat data_format = DataFormat::Float16_b;

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

// read a ublock of tiles from src to CB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        uint32_t l1_write_addr = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
        noc_async_read_tile(i, s, l1_write_addr);
        noc_async_read_barrier();
        eth_send_bytes(
            eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
            eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
            tile_bytes);
        eth_wait_for_receiver_done();
    }
}

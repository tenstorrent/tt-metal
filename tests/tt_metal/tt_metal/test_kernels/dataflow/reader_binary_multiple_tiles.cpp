// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr uint32_t single_tile_size = 2 * 1024;

    const InterleavedAddrGenFast<true> s0 = {
        .bank_base_address = src0_addr, .page_size = single_tile_size, .data_format = DataFormat::Float16_b};

    const InterleavedAddrGenFast<true> s1 = {
        .bank_base_address = src1_addr, .page_size = single_tile_size, .data_format = DataFormat::Float16_b};

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(tile_idx, s0, l1_write_addr_in0);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);

        cb_reserve_back(cb_id_in1, 1);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read_tile(tile_idx, s1, l1_write_addr_in1);
        noc_async_read_barrier();
        cb_push_back(cb_id_in1, 1);
    }
}
